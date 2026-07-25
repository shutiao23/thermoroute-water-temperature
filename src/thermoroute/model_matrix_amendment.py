"""Outcome-free Route-A model-matrix amendment contract.

This module validates immutable prelabel governance inputs, the model-matrix
amendment, and the exact contract and Git lineage of its separately committed
future seal.  It does not import training code, inspect outcome or prediction
artifacts, or use a network.  Seal construction and validation never create the
seal path; publication remains a separate, later commit performed by a caller.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any


AMENDMENT_FORMAT = "thermoroute.route-a-model-matrix-amendment.v1"
AMENDMENT_ID = "route-a-prelabel-model-matrix-replication-017"
AMENDMENT_STATUS = "FROZEN_PRELABEL_OUTCOME_FREE"
RECORDED_DATE = "2026-07-25"
AMENDMENT_RELATIVE = "protocols/route_a_model_matrix_amendment_v1.json"
AMENDMENT_SEAL_RELATIVE = (
    "protocols/route_a_model_matrix_amendment_seal_v1.json"
)
AMENDMENT_SEAL_FORMAT = "thermoroute.route-a-model-matrix-amendment-seal.v1"
AMENDMENT_SEAL_STATUS = "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
MODEL_MATRIX_CONTRACT_FORMAT = "thermoroute.route-a-model-matrix-contract.v1"
MODEL_MATRIX_SUITE_BINDING_FORMAT = (
    "thermoroute.route-a-model-matrix-suite-binding.v1"
)

HISTORY_CONTRACT = {
    "governance_seal_commits_must_be_strict_ancestors": True,
    "amendment_blob_must_match_document_commit": True,
    "amendment_document_created_exactly_once": True,
    "document_commit_must_precede_seal_commit": True,
    "seal_created_exactly_once": True,
    "amendment_and_seal_immutable_to_release_tip": True,
}

GOVERNANCE_SEALS: dict[str, tuple[str, str]] = {
    "base_protocol_seal": (
        "protocols/route_a_protocol_seal_v1.json",
        "df700fe2fac170d0466cbcaa0fae25efb4748ab82b44d49272c1e85ff5e060cd",
    ),
    "inference_amendment_seal_v1": (
        "protocols/route_a_inference_amendment_seal_v1.json",
        "28d636a108a2def8c4aed71c3a3627f29f3d47347cbb85e953c9268a2eada3b0",
    ),
    "inference_amendment_seal_v2": (
        "protocols/route_a_inference_amendment_seal_v2.json",
        "4b4e216f51f6d350912604a96aaa170887cbca7e952a1d097f03e54a7b99969d",
    ),
    "probability_metric_erratum_seal_v1": (
        "protocols/route_a_probability_metric_erratum_seal_v1.json",
        "f7f15d9f4d411fb502108c03e8843d7d00523254f0514a00f00a4e1c3299818d",
    ),
}

_FORBIDDEN_AMBIENT_GIT_VARIABLES = frozenset(
    {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_DISCOVERY_ACROSS_FILESYSTEM",
        "GIT_EXEC_PATH",
        "GIT_EXTERNAL_DIFF",
        "GIT_GRAFT_FILE",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_QUARANTINE_PATH",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)

# Exact byte bindings for the already-sealed governance chain.  These are
# constants rather than values copied from the amendment under validation.
GOVERNANCE_SHA256: dict[str, str] = {
    "protocols/route_a_confirmatory_protocol.md": (
        "9954e630b74e544af51ae17937a28ac9f2f5375c21de6dcaa6b8735502a1fa29"
    ),
    "protocols/route_a_confirmatory_v1.json": (
        "93c32e9dbfe976eaa7ef31cc5181ae1f4a415ad2b2e30674a5df64c46b968c05"
    ),
    "protocols/route_a_protocol_seal_v1.json": (
        "df700fe2fac170d0466cbcaa0fae25efb4748ab82b44d49272c1e85ff5e060cd"
    ),
    "protocols/route_a_inference_amendment_v1.json": (
        "3363e007770addee26fdab58052bfc483c00df3c52e49ddfab3d3314cfde1fff"
    ),
    "protocols/route_a_inference_amendment_seal_v1.json": (
        "28d636a108a2def8c4aed71c3a3627f29f3d47347cbb85e953c9268a2eada3b0"
    ),
    "protocols/route_a_inference_amendment_v2.json": (
        "c936dac301e7f90b05e40bfcb40f86e3a2cd88e692e422fc21fa51fee785bcf8"
    ),
    "protocols/route_a_inference_amendment_seal_v2.json": (
        "4b4e216f51f6d350912604a96aaa170887cbca7e952a1d097f03e54a7b99969d"
    ),
    "protocols/route_a_probability_metric_erratum_v1.json": (
        "d549359f77f58c6f82b5cfbf5310be3a6165dcfb83240376ad8e210b8cb5bc82"
    ),
    "protocols/route_a_probability_metric_erratum_seal_v1.json": (
        "f7f15d9f4d411fb502108c03e8843d7d00523254f0514a00f00a4e1c3299818d"
    ),
    "protocols/route_a_outcome_qc_policy_v1.json": (
        "e8fb0b6857e2c4ce76fc37704ebce3d8b821585362a3a1dd4fe980edbe059b96"
    ),
    "protocols/route_a_temporal_coverage_policy_v1.json": (
        "6b08850ced16de6f97ceda8b16ce89b301d5c5cccb794a4427da0a3e39e211ad"
    ),
    "protocols/route_a_claim_registry_v1.json": (
        "bc3d489b6f2cfe945789a57990a15291b332ae63be6c8df0211aa2b6ff8b70b6"
    ),
}

BASE_PROTOCOL_RELATIVE = "protocols/route_a_confirmatory_v1.json"
INFERENCE_AMENDMENT_RELATIVE = "protocols/route_a_inference_amendment_v2.json"

# Each digest is independently recomputed from the named JSON object.  Whole-
# file hashes above and object hashes here are intentionally separate checks.
PROTOCOL_OBJECT_SHA256: dict[str, tuple[str, tuple[str, ...], str]] = {
    "base_primary_inference_contract": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract",),
        "d2fe818ddbd2c772a6d6eeae6495c577740b271a4a5c5ebc94d5e152062fd976",
    ),
    "base_primary_estimand": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "primary_estimand"),
        "3bc667ed2b08488a2bd7fa0f19966383d06e444bbb1e2150957337a7c484a2ce",
    ),
    "base_primary_models": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "primary_models"),
        "358e9daf5ab555dbb69eec8ab7990f78e5f7a0f00c0cd07ee8fe5591335cd47a",
    ),
    "base_feature_order": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "feature_order"),
        "be54c8189ec97bf497d5931316adb89e33a01badb0e1b6f26a398234b43118ad",
    ),
    "base_mandatory_architecture_controls": (
        BASE_PROTOCOL_RELATIVE,
        (
            "primary_inference_contract",
            "mandatory_exploratory_architecture_controls",
        ),
        "5f923e976798070e99f2272391b7d2c7b2ed859d4fecf4a955a0497a90660f8b",
    ),
    "base_confirmatory_family": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "confirmatory_family"),
        "473b2ddae17b353b3499413ecd266e684e9b411412bba5463435e8400aedb799",
    ),
    "base_forecast_key": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "forecast_key"),
        "5317ed940523d471bad09dca8ecfd7e656c575a372f032262f0650d426d38b55",
    ),
    "base_confirmatory_claim_decision_contract": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "confirmatory_claim_decision_contract"),
        "651b92013c2b4c2da7994385611c03e975c879eddd84b5dd442a7fb011cf52e9",
    ),
    "base_probabilistic_event_contract": (
        BASE_PROTOCOL_RELATIVE,
        ("primary_inference_contract", "probabilistic_event_contract"),
        "7d7540732aac8dc15dc9844296322b1c75537caf467bbaae18839aa37762b098",
    ),
    "inference_scientific_comparisons": (
        INFERENCE_AMENDMENT_RELATIVE,
        ("scientific_comparisons",),
        "ad734195f2b090f2da6229ae69e9422cc709adaaec8b0591e59aa39241936220",
    ),
    "inference_estimand_scope": (
        INFERENCE_AMENDMENT_RELATIVE,
        ("estimand_scope",),
        "0975d9613c8a11bb4124e788e24d05e6e3d37deb8aa8a87859f7ca7a720f7e1e",
    ),
    "inference_decision_overlay": (
        INFERENCE_AMENDMENT_RELATIVE,
        ("decision_overlay",),
        "cbecdcbd54a97db0c7a6e612a6d9ac454f6a951825c658f53d6760926bebaba6",
    ),
}

PRELABEL_ATTESTATION = {
    "post_2020_wtemp_requested_or_inspected": False,
    "confirmation_outcomes_requested_or_inspected": False,
    "confirmation_outcome_artifact_present": False,
    "outcome_endpoint_called": False,
    "outcome_independent": True,
    "network_used": False,
}

STAGE09_CONTROLS = (
    "DampedPriorOnly",
    "TR-noDynamicPrior",
    "TR-fixedKappa",
    "TR-noRouter",
    "TR-noMoE",
    "TR-noTCN",
    "TR-unbounded",
)
STAGE09B_ARMS = (
    "PlainMLP-7var",
    "PlainCausalTCN-7var",
    "ThermoRoute-ladder-01_WTEMP",
    "ThermoRoute-ladder-02_plus_FLOW",
    "ThermoRoute-ladder-03_plus_TEMP",
    "ThermoRoute-ladder-04_plus_PRCP",
    "ThermoRoute-ladder-05_plus_RHMEAN",
    "ThermoRoute-ladder-06_plus_DH",
    "ThermoRoute-ladder-07_plus_WDSP",
)
SEEDS = (0, 1, 2, 3, 4)
SPLITS = ("val", "calib", "test")
HORIZONS = (1, 3, 7)
STAGE09B_COMPARISONS = (
    (
        "full_vs_control",
        "ThermoRoute-ladder-07_plus_WDSP-minus-PlainMLP-7var",
        "ThermoRoute-ladder-07_plus_WDSP",
        "PlainMLP-7var",
    ),
    (
        "full_vs_control",
        "ThermoRoute-ladder-07_plus_WDSP-minus-PlainCausalTCN-7var",
        "ThermoRoute-ladder-07_plus_WDSP",
        "PlainCausalTCN-7var",
    ),
    *tuple(
        (
            "adjacent_feature_ladder",
            f"{candidate}-minus-{reference}",
            candidate,
            reference,
        )
        for reference, candidate in zip(
            STAGE09B_ARMS[2:-1], STAGE09B_ARMS[3:], strict=True
        )
    ),
)

PLAIN_CONTROL_ALLOWED_INPUTS = (
    "X",
    "Mask",
    "station",
    "wtemp_t",
    "clim_t",
    "clim_tgt",
    "damped_prior",
    "phys_std",
    "logflowz",
    "season",
    "gate",
)
PLAIN_CONTROL_FORBIDDEN_INPUTS = ("y", "target_date", "wlevelz")


class ModelMatrixAmendmentError(RuntimeError):
    """The prelabel model-matrix amendment or an upstream binding is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def model_matrix_contract_id(
    stage09_architecture_control_matrix: Mapping[str, Any],
    stage09b_development_control_matrix: Mapping[str, Any],
) -> str:
    """Content-address the two frozen matrix objects with domain separation."""
    if (
        not isinstance(stage09_architecture_control_matrix, Mapping)
        or not isinstance(stage09b_development_control_matrix, Mapping)
    ):
        raise ModelMatrixAmendmentError(
            "model-matrix contract objects must be JSON mappings"
        )
    return _sha256_json(
        {
            "format": MODEL_MATRIX_CONTRACT_FORMAT,
            "stage09_architecture_control_matrix": dict(
                stage09_architecture_control_matrix
            ),
            "stage09b_development_control_matrix": dict(
                stage09b_development_control_matrix
            ),
        }
    )


def _inside(root: Path, relative: str) -> Path:
    """Return an allowlisted regular file without following symlinks."""
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or Path(relative).as_posix() != relative
        or any(part in {"", ".", ".."} for part in Path(relative).parts)
    ):
        raise ModelMatrixAmendmentError("governance path must be canonical and relative")
    root = root.resolve()
    current = root
    parts = Path(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise ModelMatrixAmendmentError(
                f"required governance input is absent: {relative}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ModelMatrixAmendmentError(
                f"governance path contains a symlink: {relative}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ModelMatrixAmendmentError(
                f"governance path crosses a non-directory: {relative}"
            )
        if index == len(parts) - 1 and not stat.S_ISREG(info.st_mode):
            raise ModelMatrixAmendmentError(
                f"governance input is not a regular file: {relative}"
            )
    return root / relative


def _require_allowlisted(path: str | Path, *, root: Path, expected: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        accepted = Path(os.path.abspath(os.fspath(root / expected)))
        supplied = Path(os.path.abspath(os.fspath(candidate)))
        if supplied != accepted:
            raise ModelMatrixAmendmentError(
                f"model-matrix amendment path is not allowlisted: {expected}"
            )
    elif candidate.as_posix() != expected:
        raise ModelMatrixAmendmentError(
            f"model-matrix amendment path is not allowlisted: {expected}"
        )
    return _inside(root, expected)


def _read_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ModelMatrixAmendmentError(f"cannot open governance input: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ModelMatrixAmendmentError(
                f"governance input is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ModelMatrixAmendmentError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json_bytes(payload: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ModelMatrixAmendmentError(f"non-finite JSON constant in {label}: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=reject_constant,
        )
    except ModelMatrixAmendmentError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelMatrixAmendmentError(f"cannot parse {label}") from exc
    if not isinstance(value, Mapping):
        raise ModelMatrixAmendmentError(f"{label} is not a JSON object")
    return value


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    return _load_json_bytes(_read_bytes(path), label=label)


def _validate_governance_inputs(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for relative, expected_sha256 in GOVERNANCE_SHA256.items():
        path = _inside(root, relative)
        actual = hashlib.sha256(_read_bytes(path)).hexdigest()
        if actual != expected_sha256:
            raise ModelMatrixAmendmentError(
                f"immutable governance checksum changed: {relative}"
            )
        bindings[Path(relative).stem] = {
            "path": relative,
            "sha256": expected_sha256,
        }
    if len(bindings) != len(GOVERNANCE_SHA256):  # stems must remain unique
        raise ModelMatrixAmendmentError("governance binding labels are not unique")
    return bindings


def _protocol_object_bindings(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for label, (relative, pointer, expected_sha256) in PROTOCOL_OBJECT_SHA256.items():
        if relative not in loaded:
            loaded[relative] = _load_json(_inside(root, relative), label=relative)
        value: Any = loaded[relative]
        for component in pointer:
            if not isinstance(value, Mapping) or component not in value:
                raise ModelMatrixAmendmentError(
                    f"protocol object is absent: {label}"
                )
            value = value[component]
        actual = _sha256_json(value)
        if actual != expected_sha256:
            raise ModelMatrixAmendmentError(
                f"immutable protocol object checksum changed: {label}"
            )
        bindings[label] = {
            "source": relative,
            "json_path": list(pointer),
            "sha256": expected_sha256,
        }
    return bindings


def _stage09_contract() -> dict[str, Any]:
    return {
        "stage": "09",
        "role": "MANDATORY_EXPLORATORY_ARCHITECTURE_CONTROLS",
        "control_arms": list(STAGE09_CONTROLS),
        "arm_count": 7,
        "seeds_per_arm": list(SEEDS),
        "seed_count_per_arm": 5,
        "expected_member_count": 35,
        "matrix_identity": "7_control_arms_x_5_fixed_seeds_equals_35_members",
        "pairing": {
            "reference": "ThermoRoute_member_with_the_exact_same_seed",
            "same_seed_required": True,
            "exact_forecast_keys_required": True,
            "exact_y_true_required": True,
            "best_seed_selection_allowed": False,
        },
        "reporting": {
            "each_seed_retained": True,
            "equal_weight_five_member_ensemble_mean_required": True,
            "causal_or_module_necessity_attribution_allowed": False,
            "capacity_matched_claim_allowed": False,
        },
    }


def _comparison_rows() -> list[dict[str, Any]]:
    return [
        {
            "comparison_family": family,
            "comparison_id": comparison_id,
            "candidate_arm_id": candidate,
            "reference_arm_id": reference,
            "seeds": list(SEEDS),
        }
        for family, comparison_id, candidate, reference in STAGE09B_COMPARISONS
    ]


def _stage09b_contract() -> dict[str, Any]:
    return {
        "stage": "09b",
        "role": "DEVELOPMENT_ONLY_EXPLORATORY_NOT_BLIND_NOT_CONFIRMATORY",
        "development_interval": ["2006-01-01", "2020-12-31"],
        "arms": list(STAGE09B_ARMS),
        "arm_count": 9,
        "seeds_per_arm": list(SEEDS),
        "seed_count_per_arm": 5,
        "expected_member_count": 45,
        "splits": list(SPLITS),
        "horizons": list(HORIZONS),
        "member_summary_cell_count": 405,
        "member_summary_cell_identity": (
            "9_arms_x_5_seeds_x_3_splits_x_3_horizons_equals_405"
        ),
        "paired_comparisons": _comparison_rows(),
        "paired_comparison_count": 8,
        "paired_seed_count": 40,
        "paired_effect_cell_count": 360,
        "paired_effect_cell_identity": (
            "8_comparisons_x_5_seeds_x_3_splits_x_3_horizons_equals_360"
        ),
        "plain_control_information_fairness_revision": {
            "affected_arms": ["PlainMLP-7var", "PlainCausalTCN-7var"],
            "change_timing": "PROSPECTIVE_PRELABEL_BEFORE_RETRAINING",
            "allowed_consumed_batch_keys": list(PLAIN_CONTROL_ALLOWED_INPUTS),
            "all_unlisted_batch_keys_forbidden_to_consume": True,
            "explicitly_forbidden_batch_keys": list(PLAIN_CONTROL_FORBIDDEN_INPUTS),
            "context_semantics": (
                "same_outcome_free_issue_time_and_derived_context_as_full_ThermoRoute"
            ),
            "common_anchor": "damped_prior",
            "prediction_head": "unrestricted_residual_added_to_damped_prior",
            "router_present": False,
            "mixture_of_experts_present": False,
            "physics_modules_present": False,
            "bounded_residual_present": False,
            "parameter_budget": {
                "reference": "full_ThermoRoute_7_variable_arm",
                "maximum_absolute_relative_difference": 0.02,
                "exact_new_parameter_count_declared_here": False,
                "exact_count_must_be_bound_by": (
                    "prospective_executable_contract_and_frozen_Stage09b_"
                    "architecture_budget_artifact_before_any_result_interpretation"
                ),
            },
            "old_stage09b_artifacts_eligible_for_final_freeze": False,
            "full_retraining_and_replay_required": True,
            "outcome_or_result_based_retuning_allowed": False,
        },
        "interpretation_boundary": {
            "historical_tuning_budget_equalized": False,
            "feature_ladder_order": [
                "WTEMP",
                "FLOW",
                "TEMP",
                "PRCP",
                "RHMEAN",
                "DH",
                "WDSP",
            ],
            "adjacent_ladder_effect_is_path_dependent": True,
            "independent_feature_importance_claim_allowed": False,
            "causal_attribution_allowed": False,
            "confirmatory_inference_allowed": False,
        },
    }


def expected_model_matrix_amendment_document(*, root: str | Path) -> dict[str, Any]:
    """Build the one exact amendment after validating every upstream byte."""
    repository = Path(root).resolve()
    governance = _validate_governance_inputs(repository)
    object_bindings = _protocol_object_bindings(repository)
    return {
        "format": AMENDMENT_FORMAT,
        "status": AMENDMENT_STATUS,
        "amendment_id": AMENDMENT_ID,
        "recorded_date": RECORDED_DATE,
        "prelabel_attestation": PRELABEL_ATTESTATION,
        "governance_inputs": governance,
        "primary_contract_object_bindings": object_bindings,
        "scientific_scope": {
            "trigger": (
                "preopening_reproducibility_and_information_fairness_audit"
            ),
            "role": "EXPLORATORY_CONTROL_EVIDENCE_COMPLETENESS_ONLY",
            "supersedes_only": (
                "the_single_seed_replication_wording_for_the_seven_mandatory_"
                "Stage09_architecture_controls_and_the_preexisting_Stage09b_"
                "plain_control_input_and_head_definition"
            ),
            "primary_model_or_feature_contract_changed": False,
            "stage09_control_identity_or_intervention_changed": False,
            "stage09b_plain_control_input_and_head_definition_changed": True,
            "stage09b_change_is_prospective_and_outcome_free": True,
            "selection_or_retuning_using_confirmation_outcomes_allowed": False,
        },
        "stage09_architecture_control_matrix": _stage09_contract(),
        "stage09b_development_control_matrix": _stage09b_contract(),
        "unchanged_primary_boundary": {
            "confirmatory_family_count": 5,
            "formal_comparisons_changed": False,
            "formal_margins_changed": False,
            "formal_estimand_changed": False,
            "formal_decisions_changed": False,
            "multiplicity_family_changed": False,
            "primary_models_changed": False,
            "primary_feature_order_changed": False,
            "primary_fit_data_changed": False,
            "primary_frozen_hyperparameters_changed": False,
            "primary_forecast_key_registry_changed": False,
            "probabilistic_event_contract_changed": False,
            "confirmation_data_may_be_used_for_fit_or_selection": False,
            "architecture_controls_remain_exploratory_descriptive": True,
            "stage09b_remains_development_only_exploratory": True,
        },
        "lineage_contract": {
            "existing_governance_files_remain_immutable": True,
            "separate_amendment_seal_required": True,
            "seal_path": AMENDMENT_SEAL_RELATIVE,
            "seal_sha256_declared_in_this_document": False,
            "amendment_document_commit_must_precede_seal_commit": True,
            "amendment_document_created_exactly_once": True,
            "seal_created_exactly_once": True,
            "document_and_seal_immutable_after_sealing": True,
        },
    }


def validate_model_matrix_amendment(
    path: str | Path,
    *,
    root: str | Path,
) -> Mapping[str, Any]:
    """Validate the exact document and independently replay all bindings."""
    repository = Path(root).resolve()
    amendment_path = _require_allowlisted(
        path,
        root=repository,
        expected=AMENDMENT_RELATIVE,
    )
    actual = _load_json(amendment_path, label="model-matrix amendment")
    expected = expected_model_matrix_amendment_document(root=repository)
    if actual != expected:
        raise ModelMatrixAmendmentError(
            "model-matrix amendment schema or content is stale/tampered"
        )
    if re.fullmatch(r"[0-9a-f]{64}", _sha256_json(actual)) is None:  # pragma: no cover
        raise ModelMatrixAmendmentError("canonical amendment digest is malformed")
    return actual


def _binding_for_relative(root: Path, relative: str) -> dict[str, str]:
    path = _inside(root, relative)
    return {
        "path": relative,
        "sha256": hashlib.sha256(_read_bytes(path)).hexdigest(),
    }


def _seal_contract(root: Path, amendment_document_commit: str) -> dict[str, Any]:
    """Return exact future-seal bytes without making a lineage claim."""
    return {
        "format": AMENDMENT_SEAL_FORMAT,
        "status": AMENDMENT_SEAL_STATUS,
        "amendment_id": AMENDMENT_ID,
        "amendment": _binding_for_relative(root, AMENDMENT_RELATIVE),
        "governance_seals": {
            label: {"path": relative, "sha256": digest}
            for label, (relative, digest) in GOVERNANCE_SEALS.items()
        },
        "amendment_document_commit": amendment_document_commit,
        "history_contract": dict(HISTORY_CONTRACT),
        "prelabel_attestation": dict(PRELABEL_ATTESTATION),
    }


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ModelMatrixAmendmentError(f"cannot audit {label} path") from exc
        if stat.S_ISLNK(mode):
            raise ModelMatrixAmendmentError(
                f"{label} path contains a symlink component"
            )


def _absolute_safe_root(root: str | Path) -> Path:
    root_path = Path(os.path.abspath(os.fspath(root)))
    if not root_path.is_dir():
        raise ModelMatrixAmendmentError("model-matrix amendment root is absent")
    _assert_no_symlink_components(root_path, label="model-matrix amendment root")
    return root_path


def _read_path_marker(path: Path, *, prefix: str, label: str) -> Path:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelMatrixAmendmentError(f"cannot read {label}") from exc
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise ModelMatrixAmendmentError(f"{label} is malformed")
    raw = lines[0][len(prefix):]
    if not raw or raw != raw.strip() or "\x00" in raw:
        raise ModelMatrixAmendmentError(f"{label} is malformed")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _assert_no_symlink_components(candidate, label=label)
    if not candidate.is_dir():
        raise ModelMatrixAmendmentError(f"{label} target is absent")
    return candidate


def _declared_git_directories(root: Path) -> tuple[Path, Path]:
    marker = root / ".git"
    try:
        marker_mode = marker.lstat().st_mode
    except (FileNotFoundError, OSError) as exc:
        raise ModelMatrixAmendmentError(
            "model-matrix lineage requires a local safe .git marker"
        ) from exc
    if stat.S_ISLNK(marker_mode):
        raise ModelMatrixAmendmentError(
            "model-matrix lineage requires a local safe .git marker"
        )
    if stat.S_ISDIR(marker_mode):
        git_directory = marker
    elif stat.S_ISREG(marker_mode):
        git_directory = _read_path_marker(
            marker,
            prefix="gitdir: ",
            label="model-matrix gitfile",
        )
    else:
        raise ModelMatrixAmendmentError(
            "model-matrix lineage requires a local safe .git marker"
        )
    common_marker = git_directory / "commondir"
    if os.path.lexists(common_marker):
        mode = common_marker.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ModelMatrixAmendmentError(
                "model-matrix commondir marker is unsafe"
            )
        common_directory = _read_path_marker(
            common_marker,
            prefix="",
            label="model-matrix commondir marker",
        )
    else:
        common_directory = git_directory
    return git_directory, common_directory


def _safe_git_environment() -> dict[str, str]:
    forbidden = sorted(
        name
        for name, value in os.environ.items()
        if name in _FORBIDDEN_AMBIENT_GIT_VARIABLES
        or name.startswith("GIT_CONFIG_KEY_")
        or name.startswith("GIT_CONFIG_VALUE_")
        or (name == "GIT_NO_REPLACE_OBJECTS" and value != "1")
    )
    if forbidden:
        raise ModelMatrixAmendmentError(
            f"ambient Git override is prohibited: {forbidden}"
        )
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-c",
                "core.useReplaceRefs=false",
                "-C",
                str(root),
                *arguments,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
            check=False,
        )
    except OSError as exc:  # pragma: no cover - Git is a formal dependency
        raise ModelMatrixAmendmentError(
            "Git is required for model-matrix lineage"
        ) from exc


def _decoded_absolute_git_path(
    result: subprocess.CompletedProcess[bytes],
    *,
    label: str,
) -> Path:
    if result.returncode:
        raise ModelMatrixAmendmentError(f"cannot resolve {label}")
    try:
        raw = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ModelMatrixAmendmentError(f"{label} is malformed") from exc
    candidate = Path(raw)
    if not raw or not candidate.is_absolute():
        raise ModelMatrixAmendmentError(f"{label} is not absolute")
    return Path(os.path.abspath(os.fspath(candidate)))


def _require_git_root(root: str | Path) -> Path:
    root_path = _absolute_safe_root(root)
    declared_git, declared_common = _declared_git_directories(root_path)
    top = _decoded_absolute_git_path(
        _git(root_path, "rev-parse", "--show-toplevel"),
        label="model-matrix Git top-level",
    )
    if top != root_path:
        raise ModelMatrixAmendmentError(
            "model-matrix root must be the exact Git top-level"
        )
    reported_git = _decoded_absolute_git_path(
        _git(root_path, "rev-parse", "--path-format=absolute", "--git-dir"),
        label="model-matrix Git directory",
    )
    reported_common = _decoded_absolute_git_path(
        _git(root_path, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        label="model-matrix Git common directory",
    )
    for reported, declared, label in (
        (reported_git, declared_git, "model-matrix Git directory"),
        (reported_common, declared_common, "model-matrix Git common directory"),
    ):
        _assert_no_symlink_components(reported, label=label)
        if not reported.is_dir() or reported != declared:
            raise ModelMatrixAmendmentError(f"{label} differs from its local marker")
    shallow = _git(root_path, "rev-parse", "--is-shallow-repository")
    if shallow.returncode or shallow.stdout.strip() != b"false":
        raise ModelMatrixAmendmentError(
            "model-matrix lineage prohibits a shallow repository"
        )
    replacements = _git(
        root_path,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    )
    if replacements.returncode or replacements.stdout.strip():
        raise ModelMatrixAmendmentError(
            "model-matrix lineage prohibits Git replacement refs"
        )
    for label, relative in (
        ("legacy grafts", "info/grafts"),
        ("object alternates", "objects/info/alternates"),
    ):
        result = _git(
            root_path,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            relative,
        )
        candidate = _decoded_absolute_git_path(
            result,
            label=f"model-matrix Git {label}",
        )
        _assert_no_symlink_components(candidate, label=f"model-matrix Git {label}")
        if os.path.lexists(candidate):
            raise ModelMatrixAmendmentError(
                f"model-matrix lineage prohibits Git {label}"
            )
    return root_path


def _has_local_git(root: Path) -> bool:
    return os.path.lexists(root / ".git")


def _full_commit(value: str, *, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ModelMatrixAmendmentError(f"{label} commit is malformed")
    return value


def _rev_parse(root: Path, revision: str) -> str:
    result = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if result.returncode:
        raise ModelMatrixAmendmentError(f"Git commit is absent: {revision}")
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ModelMatrixAmendmentError("Git commit identity is malformed") from exc
    return _full_commit(value, label=revision)


def _git_path_exists(root: Path, commit: str, relative: str) -> bool:
    result = _git(root, "cat-file", "-e", f"{commit}:{relative}")
    if result.returncode not in {0, 1, 128}:
        raise ModelMatrixAmendmentError("cannot inspect model-matrix Git lifetime")
    return result.returncode == 0


def _git_path_creation_commits(root: Path, tip: str, relative: str) -> list[str]:
    history = _git(root, "rev-list", "--reverse", "--parents", tip)
    if history.returncode:
        raise ModelMatrixAmendmentError("cannot enumerate model-matrix Git history")
    try:
        lines = history.stdout.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ModelMatrixAmendmentError("Git history is malformed") from exc
    creations: list[str] = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        commit, parents = fields[0], fields[1:]
        if not _git_path_exists(root, commit, relative):
            continue
        if not parents or all(
            not _git_path_exists(root, parent, relative) for parent in parents
        ):
            creations.append(commit)
    return creations


def _ancestry_path_commits(root: Path, start: str, tip: str) -> list[str]:
    history = _git(
        root,
        "rev-list",
        "--reverse",
        "--ancestry-path",
        f"{start}..{tip}",
    )
    if history.returncode:
        raise ModelMatrixAmendmentError("cannot replay model-matrix descendants")
    try:
        return [
            line
            for line in history.stdout.decode("ascii", errors="strict").splitlines()
            if line
        ]
    except UnicodeDecodeError as exc:
        raise ModelMatrixAmendmentError("Git history is malformed") from exc


def _require_ancestor(
    root: Path,
    ancestor: str,
    descendant: str,
    *,
    label: str,
    strict: bool = False,
) -> None:
    if strict and ancestor == descendant:
        raise ModelMatrixAmendmentError(
            f"model-matrix Git chronology is not strict: {label}"
        )
    result = _git(root, "merge-base", "--is-ancestor", ancestor, descendant)
    if result.returncode:
        raise ModelMatrixAmendmentError(
            f"model-matrix Git chronology failed: {label}"
        )


def _require_path_immutable(
    *,
    root: Path,
    start: str,
    tip: str,
    relative: str,
    expected_sha256: str,
) -> None:
    for commit in [start, *_ancestry_path_commits(root, start, tip)]:
        blob = _git(root, "show", f"{commit}:{relative}")
        if (
            blob.returncode
            or hashlib.sha256(blob.stdout).hexdigest() != expected_sha256
        ):
            raise ModelMatrixAmendmentError(
                f"model-matrix path changed after freezing: {relative}"
            )


def _validate_document_git_lineage(
    *,
    root: Path,
    amendment_document_commit: str,
    tip: str,
    amendment_sha256: str,
    require_strict_tip: bool,
) -> None:
    root = _require_git_root(root)
    document_commit = _full_commit(
        amendment_document_commit,
        label="amendment document",
    )
    tip_commit = _rev_parse(root, tip)
    if _git(root, "cat-file", "-e", f"{document_commit}^{{commit}}").returncode:
        raise ModelMatrixAmendmentError("amendment document commit is absent")
    _require_ancestor(
        root,
        document_commit,
        tip_commit,
        label="amendment-document-to-tip",
        strict=require_strict_tip,
    )
    births = _git_path_creation_commits(root, tip_commit, AMENDMENT_RELATIVE)
    if births != [document_commit]:
        raise ModelMatrixAmendmentError(
            "amendment document must have exactly one Git creation equal to its "
            "declared document commit"
        )
    _require_path_immutable(
        root=root,
        start=document_commit,
        tip=tip_commit,
        relative=AMENDMENT_RELATIVE,
        expected_sha256=amendment_sha256,
    )
    if _git_path_exists(root, document_commit, AMENDMENT_SEAL_RELATIVE):
        raise ModelMatrixAmendmentError(
            "model-matrix seal existed at the amendment document commit"
        )
    for label, (relative, expected_sha256) in GOVERNANCE_SEALS.items():
        births = _git_path_creation_commits(root, document_commit, relative)
        if len(births) != 1:
            raise ModelMatrixAmendmentError(
                f"governance seal does not have one Git creation: {label}"
            )
        birth = births[0]
        _require_ancestor(
            root,
            birth,
            document_commit,
            label=f"{label}-to-amendment-document",
            strict=True,
        )
        _require_path_immutable(
            root=root,
            start=birth,
            tip=document_commit,
            relative=relative,
            expected_sha256=expected_sha256,
        )


def _validate_model_matrix_amendment_seal_git_lineage(
    *,
    root: Path,
    amendment_document_commit: str,
    tip: str,
    amendment_sha256: str,
    seal_sha256: str,
) -> str:
    """Return the unique seal-birth commit after proving immutable history."""
    root = _require_git_root(root)
    tip_commit = _rev_parse(root, tip)
    _validate_document_git_lineage(
        root=root,
        amendment_document_commit=amendment_document_commit,
        tip=tip_commit,
        amendment_sha256=amendment_sha256,
        require_strict_tip=True,
    )
    document_commit = _full_commit(
        amendment_document_commit,
        label="amendment document",
    )
    creations = _git_path_creation_commits(root, tip_commit, AMENDMENT_SEAL_RELATIVE)
    if len(creations) != 1:
        raise ModelMatrixAmendmentError(
            "model-matrix seal must have exactly one Git creation"
        )
    creation = creations[0]
    _require_ancestor(
        root,
        document_commit,
        creation,
        label="amendment-document-to-seal",
        strict=True,
    )
    _require_ancestor(root, creation, tip_commit, label="seal-to-tip")
    _require_path_immutable(
        root=root,
        start=creation,
        tip=tip_commit,
        relative=AMENDMENT_SEAL_RELATIVE,
        expected_sha256=seal_sha256,
    )
    return creation


def _validate_optional_descendants(
    *,
    root: Path,
    seal_commit: str,
    tip: str,
    model_freeze_commit: str | None,
    authorization_commit: str | None,
    release_commit: str | None,
) -> None:
    previous = seal_commit
    for label, value in (
        ("model freeze", model_freeze_commit),
        ("authorization", authorization_commit),
        ("release", release_commit),
    ):
        if value is None:
            continue
        commit = _full_commit(value, label=label)
        _rev_parse(root, commit)
        _require_ancestor(
            root,
            previous,
            commit,
            label=f"{previous}-to-{label}",
            strict=True,
        )
        previous = commit
    _require_ancestor(root, previous, _rev_parse(root, tip), label="evidence-to-tip")


def build_model_matrix_amendment_seal_document(
    *,
    root: str | Path,
    amendment_document_commit: str,
    amendment_path: str | Path = AMENDMENT_RELATIVE,
) -> dict[str, Any]:
    """Build future seal content only after a committed, immutable document."""
    root_path = _require_git_root(root)
    validate_model_matrix_amendment(amendment_path, root=root_path)
    document_commit = _full_commit(
        amendment_document_commit,
        label="amendment document",
    )
    head = _rev_parse(root_path, "HEAD")
    amendment_sha256 = hashlib.sha256(
        _read_bytes(_inside(root_path, AMENDMENT_RELATIVE))
    ).hexdigest()
    _validate_document_git_lineage(
        root=root_path,
        amendment_document_commit=document_commit,
        tip=head,
        amendment_sha256=amendment_sha256,
        require_strict_tip=False,
    )
    seal_path = root_path / AMENDMENT_SEAL_RELATIVE
    _assert_no_symlink_components(seal_path, label="model-matrix seal")
    if os.path.lexists(seal_path):
        raise ModelMatrixAmendmentError(
            "future model-matrix seal path already exists"
        )
    if _git_path_creation_commits(root_path, head, AMENDMENT_SEAL_RELATIVE):
        raise ModelMatrixAmendmentError(
            "model-matrix history already contains a seal creation"
        )
    return _seal_contract(root_path, document_commit)


def _validate_external_digest(
    value: str | None,
    *,
    actual: str,
    label: str,
    required: bool,
) -> None:
    if value is None:
        if required:
            raise ModelMatrixAmendmentError(
                f"gitless validation requires outer-frozen {label} SHA-256"
            )
        return
    if re.fullmatch(r"[0-9a-f]{64}", value) is None or value != actual:
        raise ModelMatrixAmendmentError(f"outer-frozen {label} SHA-256 differs")


def validate_model_matrix_amendment_seal(
    seal_path: str | Path,
    *,
    root: str | Path,
    amendment_path: str | Path = AMENDMENT_RELATIVE,
    allow_gitless_archive: bool = False,
    expected_amendment_sha256: str | None = None,
    expected_seal_sha256: str | None = None,
    model_freeze_commit: str | None = None,
    authorization_commit: str | None = None,
    release_commit: str | None = None,
) -> Mapping[str, Any]:
    """Validate exact seal content and replay Git or outer-frozen bytes.

    Gitless mode proves only that the two local byte streams equal SHA-256
    digests supplied by an outer verified archive.  It never claims to prove
    commit ancestry, unique creation, or immutability history.
    """
    candidate_root = _absolute_safe_root(root)
    has_local_git = _has_local_git(candidate_root)
    root_path = _require_git_root(candidate_root) if has_local_git else candidate_root
    seal_file = _require_allowlisted(
        seal_path,
        root=root_path,
        expected=AMENDMENT_SEAL_RELATIVE,
    )
    validate_model_matrix_amendment(amendment_path, root=root_path)
    seal = _load_json(seal_file, label="model-matrix amendment seal")
    required = {
        "format",
        "status",
        "amendment_id",
        "amendment",
        "governance_seals",
        "amendment_document_commit",
        "history_contract",
        "prelabel_attestation",
    }
    commit = str(seal.get("amendment_document_commit", ""))
    if (
        set(seal) != required
        or seal.get("format") != AMENDMENT_SEAL_FORMAT
        or seal.get("status") != AMENDMENT_SEAL_STATUS
        or seal.get("amendment_id") != AMENDMENT_ID
        or seal.get("history_contract") != HISTORY_CONTRACT
        or seal.get("prelabel_attestation") != PRELABEL_ATTESTATION
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
    ):
        raise ModelMatrixAmendmentError(
            "model-matrix amendment seal schema or contract changed"
        )
    amendment_sha256 = hashlib.sha256(
        _read_bytes(_inside(root_path, AMENDMENT_RELATIVE))
    ).hexdigest()
    if seal.get("amendment") != {
        "path": AMENDMENT_RELATIVE,
        "sha256": amendment_sha256,
    }:
        raise ModelMatrixAmendmentError(
            "model-matrix seal amendment binding changed"
        )
    expected_governance = {
        label: {"path": relative, "sha256": digest}
        for label, (relative, digest) in GOVERNANCE_SEALS.items()
    }
    if seal.get("governance_seals") != expected_governance:
        raise ModelMatrixAmendmentError(
            "model-matrix seal governance registry changed"
        )
    for relative, expected_sha256 in GOVERNANCE_SEALS.values():
        actual = hashlib.sha256(_read_bytes(_inside(root_path, relative))).hexdigest()
        if actual != expected_sha256:
            raise ModelMatrixAmendmentError(
                f"model-matrix governance seal checksum changed: {relative}"
            )
    seal_sha256 = hashlib.sha256(_read_bytes(seal_file)).hexdigest()
    _validate_external_digest(
        expected_amendment_sha256,
        actual=amendment_sha256,
        label="amendment",
        required=not has_local_git,
    )
    _validate_external_digest(
        expected_seal_sha256,
        actual=seal_sha256,
        label="seal",
        required=not has_local_git,
    )
    if has_local_git:
        seal_commit = _validate_model_matrix_amendment_seal_git_lineage(
            root=root_path,
            amendment_document_commit=commit,
            tip="HEAD",
            amendment_sha256=amendment_sha256,
            seal_sha256=seal_sha256,
        )
        _validate_optional_descendants(
            root=root_path,
            seal_commit=seal_commit,
            tip="HEAD",
            model_freeze_commit=model_freeze_commit,
            authorization_commit=authorization_commit,
            release_commit=release_commit,
        )
    else:
        if not allow_gitless_archive:
            raise ModelMatrixAmendmentError(
                "model-matrix amendment lineage requires Git outside archive replay"
            )
        if any(
            value is not None
            for value in (model_freeze_commit, authorization_commit, release_commit)
        ):
            raise ModelMatrixAmendmentError(
                "gitless validation cannot prove optional descendant commits"
            )
    return seal


__all__ = [
    "AMENDMENT_FORMAT",
    "AMENDMENT_ID",
    "AMENDMENT_RELATIVE",
    "AMENDMENT_SEAL_RELATIVE",
    "AMENDMENT_SEAL_FORMAT",
    "AMENDMENT_SEAL_STATUS",
    "AMENDMENT_STATUS",
    "GOVERNANCE_SHA256",
    "GOVERNANCE_SEALS",
    "HISTORY_CONTRACT",
    "ModelMatrixAmendmentError",
    "MODEL_MATRIX_CONTRACT_FORMAT",
    "MODEL_MATRIX_SUITE_BINDING_FORMAT",
    "PLAIN_CONTROL_ALLOWED_INPUTS",
    "PLAIN_CONTROL_FORBIDDEN_INPUTS",
    "PROTOCOL_OBJECT_SHA256",
    "RECORDED_DATE",
    "build_model_matrix_amendment_seal_document",
    "expected_model_matrix_amendment_document",
    "model_matrix_contract_id",
    "validate_model_matrix_amendment",
    "validate_model_matrix_amendment_seal",
]
