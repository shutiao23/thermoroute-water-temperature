"""Outcome-free governance for the Route-A probability-metric erratum.

The erratum separates two forecast objects that must not be conflated:
post-CQR endpoints are used for interval coverage and width, while the frozen
pre-CQR q05/q50/q95 heads are used for pinball loss.  This module reads only
prelabel governance documents and Git metadata.  It never accepts confirmation
outcomes, predictions, metric values, or network resources.

The document and its seal deliberately have separate lifetimes.  The erratum
must first exist in a committed tree.  A later commit may create the seal once,
and both paths must then remain byte-identical through the release tip.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping


ERRATUM_FORMAT = "thermoroute.route-a-probability-metric-erratum.v1"
ERRATUM_SEAL_FORMAT = "thermoroute.route-a-probability-metric-erratum-seal.v1"
ERRATUM_ID = "route-a-prelabel-probability-metric-semantics-016"
ERRATUM_STATUS = "FROZEN_PRELABEL_OUTCOME_FREE"
ERRATUM_SEAL_STATUS = "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"

BASE_PROTOCOL_RELATIVE = "protocols/route_a_confirmatory_v1.json"
BASE_PROTOCOL_SEAL_RELATIVE = "protocols/route_a_protocol_seal_v1.json"
INFERENCE_AMENDMENT_RELATIVE = "protocols/route_a_inference_amendment_v2.json"
INFERENCE_AMENDMENT_SEAL_RELATIVE = "protocols/route_a_inference_amendment_seal_v2.json"
ERRATUM_RELATIVE = "protocols/route_a_probability_metric_erratum_v1.json"
ERRATUM_SEAL_RELATIVE = "protocols/route_a_probability_metric_erratum_seal_v1.json"

PRELABEL_ATTESTATION = {
    "post_2020_wtemp_requested_or_inspected": False,
    "confirmation_outcomes_requested_or_inspected": False,
    "outcome_endpoint_called": False,
    "outcome_independent": True,
    "network_used": False,
}

HISTORY_CONTRACT = {
    "governance_seal_commits_must_be_ancestors": True,
    "erratum_blob_must_match_document_commit": True,
    "erratum_document_created_exactly_once": True,
    "document_commit_must_precede_seal_commit": True,
    "seal_created_exactly_once": True,
    "erratum_and_seal_immutable_to_release_tip": True,
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
        "GIT_GLOB_PATHSPECS",
        "GIT_GRAFT_FILE",
        "GIT_ICASE_PATHSPECS",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_LITERAL_PATHSPECS",
        "GIT_NAMESPACE",
        "GIT_NOGLOB_PATHSPECS",
        "GIT_OBJECT_DIRECTORY",
        "GIT_PREFIX",
        "GIT_QUARANTINE_PATH",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
)


class ProbabilityMetricErratumError(RuntimeError):
    """The metric erratum or its immutable lineage is absent or inconsistent."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str, *, require_file: bool = True) -> Path:
    """Resolve one canonical repository path without following symlinks."""
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
        or Path(relative).as_posix() != relative
        or any(part in {".", ".."} for part in Path(relative).parts)
    ):
        raise ProbabilityMetricErratumError("probability-erratum path must be repository-relative")
    root = root.resolve()
    path = root / relative
    current = root
    parts = Path(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            current.lstat()
        except FileNotFoundError:
            if require_file or index < len(parts) - 1:
                raise ProbabilityMetricErratumError(
                    f"required probability-erratum input is absent: {relative}"
                )
            return path
        if current.is_symlink():
            raise ProbabilityMetricErratumError(
                f"probability-erratum path contains a symlink: {relative}"
            )
        if index < len(parts) - 1 and not current.is_dir():
            raise ProbabilityMetricErratumError(
                f"probability-erratum path crosses a non-directory: {relative}"
            )
        if index == len(parts) - 1 and require_file and not current.is_file():
            raise ProbabilityMetricErratumError(
                f"required probability-erratum input is not a file: {relative}"
            )
    if require_file and not path.is_file():
        raise ProbabilityMetricErratumError(
            f"required probability-erratum input is absent: {relative}"
        )
    return path


def _require_allowlisted(
    path: str | Path,
    *,
    root: Path,
    expected: str,
) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = root / resolved
    resolved = Path(os.path.abspath(os.fspath(resolved)))
    allowed = Path(os.path.abspath(os.fspath(root / expected)))
    if resolved != allowed:
        raise ProbabilityMetricErratumError(
            f"probability-erratum input is not allowlisted: expected {expected}"
        )
    return _inside(root, expected)


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilityMetricErratumError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ProbabilityMetricErratumError(f"{label} is not a JSON object")
    return value


def _binding(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
    }


def _validate_binding(root: Path, binding: object, *, label: str) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ProbabilityMetricErratumError(f"{label} binding is malformed")
    path = _inside(root, str(binding.get("path", "")))
    digest = binding.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProbabilityMetricErratumError(f"{label} binding lacks a SHA-256")
    if _sha256_file(path) != digest:
        raise ProbabilityMetricErratumError(f"{label} checksum changed")
    return path


def _protocol_contracts(
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    inference = protocol.get("primary_inference_contract")
    if not isinstance(inference, Mapping):
        raise ProbabilityMetricErratumError("base protocol lacks its inference contract")
    family = inference.get("confirmatory_family")
    probability = inference.get("probabilistic_event_contract")
    if not isinstance(family, list) or len(family) != 5:
        raise ProbabilityMetricErratumError(
            "base protocol does not contain exactly five formal tests"
        )
    if not all(isinstance(item, Mapping) for item in family):
        raise ProbabilityMetricErratumError("base protocol confirmatory family is malformed")
    if not isinstance(probability, Mapping):
        raise ProbabilityMetricErratumError("base protocol lacks its probability contract")
    return [dict(item) for item in family], probability


def probability_metric_erratum_contract(
    *,
    confirmatory_family_sha256: str,
    probabilistic_event_contract_sha256: str,
) -> dict[str, Any]:
    """Return the exact outcome-free scientific correction."""
    return {
        "scientific_scope": {
            "role": "DESCRIPTIVE_NOT_IN_CONFIRMATORY_FIVE_TEST_FAMILY",
            "confirmatory_family_count": 5,
            "confirmatory_family_sha256": confirmatory_family_sha256,
            "formal_comparisons_changed": False,
            "formal_margins_changed": False,
            "formal_decisions_changed": False,
            "probabilistic_event_contract_sha256": (probabilistic_event_contract_sha256),
            "inference_allowed": False,
        },
        "defect": {
            "affected_metric": ("equal_weight_three_quantile_pinball_mean_not_CRPS"),
            "stored_q05_q95_semantics": "post_2018_CQR_interval_endpoints",
            "invalid_use": (
                "scoring_stored_post_CQR_q05_and_q95_as_nominal_0.05_and_0.95_quantiles"
            ),
            "development_stage_19_semantics": ("nominal_pre_CQR_ensemble_q05_q50_q95"),
            "correction": (
                "directly_score_the_bundle_generated_transient_nominal_pre_CQR_"
                "heads_for_pinball_while_retaining_post_CQR_endpoints_for_"
                "interval_metrics"
            ),
            "interval_metrics_affected": False,
            "event_metrics_affected": False,
        },
        "corrected_metric_contract": {
            "interval_coverage_and_width_source": ("bundle_frozen_2018_cqr_endpoints"),
            "pinball_quantile_source": (
                "direct_nominal_pre_cqr_ensemble_q05_q50_q95_retained_in_memory_before_cqr"
            ),
            "nominal_quantile_handling": {
                "source": (
                    "direct_nominal_pre_cqr_ensemble_q05_q50_q95_retained_in_memory_before_cqr"
                ),
                "storage": ("transient_in_memory_only_not_written_to_public_prediction_products"),
                "member_aggregation": "equal_weight_member_mean_before_cqr",
                "cqr_forward_parity": (
                    "bitwise_float64_nominal_q05_minus_offset_and_nominal_q95_"
                    "plus_offset_equal_stored_endpoints"
                ),
                "q50_forward_parity": ("bitwise_nominal_q50_equal_stored_q50"),
                "endpoint_inversion_used": False,
                "public_prediction_schema_changed": False,
            },
            "bundle_scoring_pipeline_contracts": {
                "LightGBM": (
                    "bundle_declared_median_preserving_endpoint_clip_per_member_"
                    "then_equal_weight_member_mean_then_frozen_cqr"
                ),
                "deep_LSTM_and_deterministic_controls": (
                    "quantiles_ordered_by_construction_per_member_then_equal_"
                    "weight_member_mean_then_frozen_cqr"
                ),
            },
            "pinball_aggregation": {
                "quantiles": [0.05, 0.5, 0.95],
                "per_quantile": "station_balanced_weighted_mean_pinball_loss",
                "across_quantiles": "unscaled_arithmetic_mean_of_three_scores",
                "called_CRPS": False,
            },
            "event_probability_source": ("bundle_frozen_2018_platt_calibrated_p_exceed"),
            "event_outcome_source": (
                "confirmation_y_true_above_bundle_frozen_development_train_q90"
            ),
            "event_probability_metrics": [
                "Brier_score",
                "Brier_skill_vs_frozen_seasonal_reference",
                "log_loss_clipped_1e-6",
                "AUROC",
                "AUPRC",
                "ECE_10_equal_width_bins",
                "evaluation_calibration_intercept",
                "evaluation_calibration_slope",
            ],
            "all_event_probability_metrics_use_post_Platt_probability": True,
            "station_weighting": ("each_reportable_station_equal_weight_within_model_horizon"),
            "event_counts": "raw_unweighted_counts_and_station_balanced_rates",
            "deterministic_builtin_status": ("NOT_AVAILABLE_NO_PROBABILISTIC_HEAD"),
            "inference_prohibited": [
                "p_value",
                "confidence_interval",
                "Holm_adjustment",
                "pass_fail_decision",
                "conditional_coverage_claim",
            ],
        },
        "unchanged_contract": {
            "model_definition_changed": False,
            "fit_data_changed": False,
            "frozen_hyperparameters_changed": False,
            "hyperparameter_retuning_allowed": False,
            "old_pre_erratum_model_artifacts_eligible_for_final_freeze": False,
            "full_retraining_and_replay_under_new_source_required": True,
            "retraining_role": (
                "mechanical_reproduction_under_the_unchanged_frozen_contract_not_retuning"
            ),
            "confirmation_data_may_change_or_reestimate_CQR": False,
            "confirmation_data_may_change_or_reestimate_Platt": False,
            "confirmation_data_may_change_or_reestimate_event_threshold": False,
            "mechanical_recomputation_contract": {
                "required": True,
                "components": ["model_fits", "CQR", "Platt", "event_threshold"],
                "fit_inputs": "identical_frozen_development_only_fit_inputs",
                "algorithm": "identical_frozen_algorithm",
                "parameters": "identical_frozen_fitting_parameters",
                "purpose": "deterministic_mechanical_reproduction_only",
                "confirmation_data_used_for_fit": False,
                "retuning_or_adaptation_allowed": False,
            },
            "five_formal_tests_changed": False,
            "formal_test_margins_changed": False,
            "formal_decisions_changed": False,
            "forecast_key_registry_changed": False,
            "minimum_targets_per_station_horizon": 100,
            "station_balancing_changed": False,
        },
        "implementation_requirements": {
            "opening_implementation": "src/thermoroute/opening.py",
            "development_reference": "scripts/19_probabilistic.py",
            "targeted_opening_tests": "tests/test_confirmatory_opening.py",
            "trusted_artifact_format": ("thermoroute.route-a-probabilistic-evaluation.v2"),
            "trusted_artifact_path": "trusted/probabilistic_evaluation_v2.json",
            "transient_nominal_quantiles_required_for_v2_replay": True,
            "bitwise_forward_cqr_parity_required": True,
            "source_fields_required_in_artifact": [
                "interval_endpoint_source",
                "pinball_quantile_source",
                "event_probability_source",
                "event_outcome_source",
            ],
        },
        "lineage_contract": {
            "separate_erratum_seal_required": True,
            "seal_path": ERRATUM_SEAL_RELATIVE,
            "erratum_document_commit_must_precede_seal_commit": True,
            "both_paths_immutable_after_sealing": True,
        },
    }


def expected_probability_metric_erratum_document(
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Build the exact erratum using only already-frozen governance inputs."""
    root_path = Path(root).resolve()
    protocol_path = _inside(root_path, BASE_PROTOCOL_RELATIVE)
    protocol_seal_path = _inside(root_path, BASE_PROTOCOL_SEAL_RELATIVE)
    amendment_path = _inside(root_path, INFERENCE_AMENDMENT_RELATIVE)
    amendment_seal_path = _inside(root_path, INFERENCE_AMENDMENT_SEAL_RELATIVE)
    protocol = _load_json(protocol_path, label="base protocol")
    protocol_seal = _load_json(protocol_seal_path, label="base protocol seal")
    amendment = _load_json(amendment_path, label="inference amendment")
    amendment_seal = _load_json(amendment_seal_path, label="inference amendment seal")
    family, probability = _protocol_contracts(protocol)

    final_protocol = protocol_seal.get("final_prelabel_protocol")
    final_protocol_json = (
        final_protocol.get("json") if isinstance(final_protocol, Mapping) else None
    )
    if (
        protocol.get("protocol_id") != "route-a-confirmatory-v1"
        or protocol_seal.get("format") != "thermoroute.route-a-protocol-seal.v1"
        or protocol_seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or final_protocol_json != _binding(root_path, protocol_path)
    ):
        raise ProbabilityMetricErratumError(
            "base protocol or its seal changed before erratum validation"
        )
    if (
        amendment.get("format") != "thermoroute.route-a-inference-amendment.v2"
        or amendment.get("status") != "FROZEN_PRELABEL_OUTCOME_FREE"
        or amendment.get("post_2020_wtemp_requested_or_inspected") is not False
        or amendment.get("outcome_independent") is not True
        or amendment_seal.get("format") != "thermoroute.route-a-inference-amendment-seal.v2"
        or amendment_seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
        or amendment_seal.get("amendment") != _binding(root_path, amendment_path)
    ):
        raise ProbabilityMetricErratumError(
            "inference amendment or its seal changed before erratum validation"
        )

    contracts = probability_metric_erratum_contract(
        confirmatory_family_sha256=_sha256_json(family),
        probabilistic_event_contract_sha256=_sha256_json(probability),
    )
    return {
        "format": ERRATUM_FORMAT,
        "status": ERRATUM_STATUS,
        "erratum_id": ERRATUM_ID,
        "recorded_date": "2026-07-24",
        "prelabel_attestation": dict(PRELABEL_ATTESTATION),
        "governance_inputs": {
            "base_protocol": _binding(root_path, protocol_path),
            "base_protocol_seal": _binding(root_path, protocol_seal_path),
            "inference_amendment": _binding(root_path, amendment_path),
            "inference_amendment_seal": _binding(root_path, amendment_seal_path),
        },
        **contracts,
    }


def validate_probability_metric_erratum(
    erratum_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Require exact schema, hashes, semantics, and outcome-free attestation."""
    root_path = Path(root).resolve()
    erratum_file = _require_allowlisted(erratum_path, root=root_path, expected=ERRATUM_RELATIVE)
    actual = _load_json(erratum_file, label="probability metric erratum")
    expected = expected_probability_metric_erratum_document(root=root_path)
    if dict(actual) != expected:
        raise ProbabilityMetricErratumError(
            "probability metric erratum is stale, malformed, or tampered"
        )
    if actual.get("prelabel_attestation") != PRELABEL_ATTESTATION:
        raise ProbabilityMetricErratumError("probability metric erratum is not outcome-free")
    return dict(actual)


def _safe_git_environment() -> dict[str, str]:
    """Return a deterministic Git environment that cannot redirect history."""
    forbidden = sorted(
        name
        for name, value in os.environ.items()
        if name in _FORBIDDEN_AMBIENT_GIT_VARIABLES
        or name.startswith("GIT_CONFIG_KEY_")
        or name.startswith("GIT_CONFIG_VALUE_")
        or (name == "GIT_NO_REPLACE_OBJECTS" and value != "1")
    )
    if forbidden:
        raise ProbabilityMetricErratumError(
            f"ambient Git repository/configuration override is prohibited: {forbidden}"
        )
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
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
    """Run every lineage query through the sole hardened Git adapter."""
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
        raise ProbabilityMetricErratumError(
            "Git is required for probability-erratum lineage"
        ) from exc


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    """Reject every existing symlink component without dereferencing it."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ProbabilityMetricErratumError(
                f"cannot audit {label} path"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ProbabilityMetricErratumError(
                f"{label} path contains a symlink component"
            )


def _absolute_safe_root(root: str | Path) -> Path:
    root_path = Path(os.path.abspath(os.fspath(root)))
    if not root_path.is_dir():
        raise ProbabilityMetricErratumError("probability-erratum root is absent")
    _assert_no_symlink_components(root_path, label="probability-erratum root")
    return root_path


def _read_single_path_marker(path: Path, *, prefix: str, label: str) -> Path:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProbabilityMetricErratumError(f"cannot read {label}") from exc
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise ProbabilityMetricErratumError(f"{label} is malformed")
    raw = lines[0][len(prefix):]
    if not raw or raw != raw.strip() or "\x00" in raw:
        raise ProbabilityMetricErratumError(f"{label} is malformed")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _assert_no_symlink_components(candidate, label=label)
    if not candidate.is_dir():
        raise ProbabilityMetricErratumError(f"{label} target is absent")
    return candidate


def _declared_git_directories(root: Path) -> tuple[Path, Path]:
    """Read local gitfile/commondir markers before allowing Git to follow them."""
    marker = root / ".git"
    try:
        marker_mode = marker.lstat().st_mode
    except FileNotFoundError:
        marker_mode = 0
    except OSError as exc:
        raise ProbabilityMetricErratumError(
            "cannot audit the local probability-erratum .git marker"
        ) from exc
    if marker_mode == 0 or stat.S_ISLNK(marker_mode):
        raise ProbabilityMetricErratumError(
            "probability-erratum Git lineage requires a local safe .git marker"
        )
    if stat.S_ISDIR(marker_mode):
        git_directory = marker
    elif stat.S_ISREG(marker_mode):
        git_directory = _read_single_path_marker(
            marker,
            prefix="gitdir: ",
            label="probability-erratum gitfile",
        )
    else:
        raise ProbabilityMetricErratumError(
            "probability-erratum Git lineage requires a local safe .git marker"
        )

    common_marker = git_directory / "commondir"
    if os.path.lexists(common_marker):
        try:
            common_mode = common_marker.lstat().st_mode
        except OSError as exc:
            raise ProbabilityMetricErratumError(
                "cannot audit probability-erratum commondir marker"
            ) from exc
        if stat.S_ISLNK(common_mode) or not stat.S_ISREG(common_mode):
            raise ProbabilityMetricErratumError(
                "probability-erratum commondir marker is unsafe"
            )
        common_directory = _read_single_path_marker(
            common_marker,
            prefix="",
            label="probability-erratum commondir marker",
        )
    else:
        common_directory = git_directory
    return git_directory, common_directory


def _decode_git_path(
    result: subprocess.CompletedProcess[bytes],
    *,
    label: str,
) -> Path:
    if result.returncode:
        raise ProbabilityMetricErratumError(f"cannot resolve {label}")
    try:
        raw = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise ProbabilityMetricErratumError(f"{label} is malformed") from exc
    path = Path(raw)
    if not raw or not path.is_absolute():
        raise ProbabilityMetricErratumError(f"{label} is not an absolute path")
    return Path(os.path.abspath(os.fspath(path)))


def _require_git_root(root: str | Path) -> Path:
    """Require one complete, unredirected Git worktree rooted exactly here."""
    root_path = _absolute_safe_root(root)
    declared_git, declared_common = _declared_git_directories(root_path)

    top_path = _decode_git_path(
        _git(root_path, "rev-parse", "--show-toplevel"),
        label="probability-erratum Git top-level",
    )
    if top_path != root_path:
        raise ProbabilityMetricErratumError(
            "probability-erratum root must be the exact Git top-level"
        )

    reported_git = _decode_git_path(
        _git(
            root_path,
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
        ),
        label="probability-erratum Git directory",
    )
    reported_common = _decode_git_path(
        _git(
            root_path,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ),
        label="probability-erratum Git common directory",
    )
    for reported, declared, label in (
        (reported_git, declared_git, "probability-erratum Git directory"),
        (
            reported_common,
            declared_common,
            "probability-erratum Git common directory",
        ),
    ):
        _assert_no_symlink_components(reported, label=label)
        if not reported.is_dir() or reported != declared:
            raise ProbabilityMetricErratumError(
                f"{label} differs from its local marker"
            )

    shallow = _git(root_path, "rev-parse", "--is-shallow-repository")
    if shallow.returncode or shallow.stdout.strip() != b"false":
        raise ProbabilityMetricErratumError(
            "probability-erratum lineage prohibits a shallow repository"
        )
    replacements = _git(
        root_path,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    )
    if replacements.returncode or replacements.stdout.strip():
        raise ProbabilityMetricErratumError(
            "probability-erratum lineage prohibits Git replacement refs"
        )
    for label, relative in (
        ("legacy grafts", "info/grafts"),
        ("object alternates", "objects/info/alternates"),
    ):
        location = _git(
            root_path,
            "rev-parse",
            "--path-format=absolute",
            "--git-path",
            relative,
        )
        candidate = _decode_git_path(
            location,
            label=f"probability-erratum Git {label}",
        )
        _assert_no_symlink_components(
            candidate,
            label=f"probability-erratum Git {label}",
        )
        if os.path.lexists(candidate):
            raise ProbabilityMetricErratumError(
                f"probability-erratum lineage prohibits Git {label}"
            )
    return root_path


def _has_local_git(root: str | Path) -> bool:
    """Do not mistake an ancestor repository for archive-local lineage."""
    root_path = Path(os.path.abspath(os.fspath(root)))
    return os.path.lexists(root_path / ".git")


def _git_path_exists(root: Path, commit: str, relative: str) -> bool:
    result = _git(root, "cat-file", "-e", f"{commit}:{relative}")
    if result.returncode not in {0, 1, 128}:
        raise ProbabilityMetricErratumError("cannot inspect probability-erratum Git lifetime")
    return result.returncode == 0


def _git_path_creation_commits(
    root: Path,
    tip: str,
    relative: str,
) -> list[str]:
    history = _git(root, "rev-list", "--reverse", "--parents", tip)
    if history.returncode:
        raise ProbabilityMetricErratumError("cannot enumerate probability-erratum Git history")
    try:
        lines = history.stdout.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ProbabilityMetricErratumError(
            "probability-erratum history contains a malformed commit"
        ) from exc
    creations: list[str] = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        commit, parents = fields[0], fields[1:]
        if not _git_path_exists(root, commit, relative):
            continue
        if not parents or all(not _git_path_exists(root, parent, relative) for parent in parents):
            creations.append(commit)
    return creations


def _git_ancestry_path_commits(
    root: Path,
    start_exclusive: str,
    end_inclusive: str,
) -> list[str]:
    history = _git(
        root,
        "rev-list",
        "--reverse",
        "--ancestry-path",
        f"{start_exclusive}..{end_inclusive}",
    )
    if history.returncode:
        raise ProbabilityMetricErratumError("cannot replay probability-erratum Git descendants")
    try:
        return [
            line for line in history.stdout.decode("ascii", errors="strict").splitlines() if line
        ]
    except UnicodeDecodeError as exc:
        raise ProbabilityMetricErratumError(
            "probability-erratum history contains a malformed commit"
        ) from exc


def _require_ancestor(
    root: Path,
    ancestor: str,
    descendant: str,
    *,
    label: str,
) -> None:
    if _git(root, "merge-base", "--is-ancestor", ancestor, descendant).returncode:
        raise ProbabilityMetricErratumError(f"probability-erratum Git chronology failed: {label}")


def _require_path_immutable(
    *,
    root: Path,
    start: str,
    tip: str,
    relative: str,
    expected_sha256: str,
) -> None:
    for commit in [start, *_git_ancestry_path_commits(root, start, tip)]:
        blob = _git(root, "show", f"{commit}:{relative}")
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected_sha256:
            raise ProbabilityMetricErratumError(
                f"probability-erratum path changed after freezing: {relative}"
            )


def _validate_probability_metric_erratum_seal_git_lineage(
    *,
    root: Path,
    erratum_document_commit: str,
    tip: str,
    expected_sha256: str,
) -> str:
    """Prove one strictly later seal birth and immutable descendant bytes."""
    root = _require_git_root(root)
    if _git_path_exists(root, erratum_document_commit, ERRATUM_SEAL_RELATIVE):
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal existed at the document commit"
        )
    creations = _git_path_creation_commits(root, tip, ERRATUM_SEAL_RELATIVE)
    if len(creations) != 1:
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal must have exactly one Git creation"
        )
    creation = creations[0]
    if creation == erratum_document_commit:
        raise ProbabilityMetricErratumError(
            "probability metric erratum document and seal share one commit"
        )
    _require_ancestor(
        root,
        erratum_document_commit,
        creation,
        label="erratum-document-to-seal",
    )
    _require_ancestor(root, creation, tip, label="erratum-seal-to-tip")
    _require_path_immutable(
        root=root,
        start=creation,
        tip=tip,
        relative=ERRATUM_SEAL_RELATIVE,
        expected_sha256=expected_sha256,
    )
    return creation


def _governance_commits(
    root: Path,
) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    base_seal = _load_json(
        _inside(root, BASE_PROTOCOL_SEAL_RELATIVE),
        label="base protocol seal",
    )
    amendment_seal = _load_json(
        _inside(root, INFERENCE_AMENDMENT_SEAL_RELATIVE),
        label="inference amendment seal",
    )
    final_protocol = base_seal.get("final_prelabel_protocol")
    base_commit = (
        str(final_protocol.get("commit", "")) if isinstance(final_protocol, Mapping) else ""
    )
    amendment_commit = str(amendment_seal.get("final_prelabel_commit", ""))
    if (
        re.fullmatch(r"[0-9a-f]{40}", base_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", amendment_commit) is None
    ):
        raise ProbabilityMetricErratumError("governance seal commit identity is malformed")
    return base_commit, amendment_commit, base_seal, amendment_seal


def _validate_document_git_lineage(
    *,
    root: Path,
    erratum_document_commit: str,
    tip: str,
    erratum_file: Path,
) -> None:
    root = _require_git_root(root)
    if _git(root, "cat-file", "-e", f"{erratum_document_commit}^{{commit}}").returncode:
        raise ProbabilityMetricErratumError("probability metric erratum document commit is absent")
    _require_ancestor(
        root,
        erratum_document_commit,
        tip,
        label="erratum-document-to-tip",
    )
    document_births = _git_path_creation_commits(root, tip, ERRATUM_RELATIVE)
    if document_births != [erratum_document_commit]:
        raise ProbabilityMetricErratumError(
            "probability metric erratum document must have exactly one Git "
            "creation equal to its declared document commit"
        )
    expected_sha256 = _sha256_file(erratum_file)
    _require_path_immutable(
        root=root,
        start=erratum_document_commit,
        tip=tip,
        relative=ERRATUM_RELATIVE,
        expected_sha256=expected_sha256,
    )
    base_commit, amendment_commit, _base_seal, _amendment_seal = _governance_commits(root)
    _require_ancestor(root, base_commit, erratum_document_commit, label="protocol-to-erratum")
    _require_ancestor(
        root,
        amendment_commit,
        erratum_document_commit,
        label="inference-amendment-to-erratum",
    )
    amendment_seal_births = _git_path_creation_commits(
        root, erratum_document_commit, INFERENCE_AMENDMENT_SEAL_RELATIVE
    )
    if len(amendment_seal_births) != 1 or amendment_seal_births[0] == erratum_document_commit:
        raise ProbabilityMetricErratumError(
            "inference amendment seal did not have one strictly pre-erratum Git birth"
        )
    _require_ancestor(
        root,
        amendment_seal_births[0],
        erratum_document_commit,
        label="inference-amendment-seal-to-erratum",
    )
    for relative in (
        BASE_PROTOCOL_SEAL_RELATIVE,
        INFERENCE_AMENDMENT_SEAL_RELATIVE,
    ):
        blob = _git(root, "show", f"{erratum_document_commit}:{relative}")
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != _sha256_file(
            _inside(root, relative)
        ):
            raise ProbabilityMetricErratumError(
                f"governance seal was not frozen before the erratum: {relative}"
            )


def build_probability_metric_erratum_seal_document(
    *,
    root: str | Path,
    erratum_document_commit: str,
    erratum_path: str | Path = ERRATUM_RELATIVE,
) -> dict[str, Any]:
    """Build the separate seal only after the erratum document is committed."""
    root_path = _require_git_root(root)
    erratum = validate_probability_metric_erratum(erratum_path, root=root_path)
    erratum_file = _inside(root_path, ERRATUM_RELATIVE)
    if re.fullmatch(r"[0-9a-f]{40}", erratum_document_commit) is None:
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal requires a full Git commit"
        )
    _validate_document_git_lineage(
        root=root_path,
        erratum_document_commit=erratum_document_commit,
        tip=erratum_document_commit,
        erratum_file=erratum_file,
    )
    if _git_path_creation_commits(root_path, erratum_document_commit, ERRATUM_SEAL_RELATIVE):
        raise ProbabilityMetricErratumError(
            "erratum document history already contains the separate seal"
        )
    return {
        "format": ERRATUM_SEAL_FORMAT,
        "status": ERRATUM_SEAL_STATUS,
        "erratum_id": erratum["erratum_id"],
        "erratum": _binding(root_path, erratum_file),
        "governance_seals": {
            "base_protocol_seal": _binding(
                root_path, _inside(root_path, BASE_PROTOCOL_SEAL_RELATIVE)
            ),
            "inference_amendment_seal": _binding(
                root_path,
                _inside(root_path, INFERENCE_AMENDMENT_SEAL_RELATIVE),
            ),
        },
        "erratum_document_commit": erratum_document_commit,
        "history_contract": dict(HISTORY_CONTRACT),
        "prelabel_attestation": dict(PRELABEL_ATTESTATION),
    }


def validate_probability_metric_erratum_seal(
    seal_path: str | Path,
    *,
    root: str | Path,
    erratum_path: str | Path = ERRATUM_RELATIVE,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Validate seal schema, bindings, attestation, and live Git lineage."""
    candidate_root = _absolute_safe_root(root)
    has_local_git = _has_local_git(candidate_root)
    root_path = (
        _require_git_root(candidate_root)
        if has_local_git
        else candidate_root
    )
    seal_file = _require_allowlisted(seal_path, root=root_path, expected=ERRATUM_SEAL_RELATIVE)
    erratum = validate_probability_metric_erratum(erratum_path, root=root_path)
    seal = _load_json(seal_file, label="probability metric erratum seal")
    required = {
        "format",
        "status",
        "erratum_id",
        "erratum",
        "governance_seals",
        "erratum_document_commit",
        "history_contract",
        "prelabel_attestation",
    }
    if (
        set(seal) != required
        or seal.get("format") != ERRATUM_SEAL_FORMAT
        or seal.get("status") != ERRATUM_SEAL_STATUS
        or seal.get("erratum_id") != erratum["erratum_id"]
        or seal.get("history_contract") != HISTORY_CONTRACT
        or seal.get("prelabel_attestation") != PRELABEL_ATTESTATION
    ):
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal schema or contract changed"
        )
    erratum_file = _validate_binding(
        root_path, seal.get("erratum"), label="probability metric erratum"
    )
    if erratum_file != _inside(root_path, ERRATUM_RELATIVE):
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal names a noncanonical document"
        )
    governance = seal.get("governance_seals")
    if not isinstance(governance, Mapping) or set(governance) != {
        "base_protocol_seal",
        "inference_amendment_seal",
    }:
        raise ProbabilityMetricErratumError(
            "probability metric erratum seal governance registry changed"
        )
    for key, relative in (
        ("base_protocol_seal", BASE_PROTOCOL_SEAL_RELATIVE),
        ("inference_amendment_seal", INFERENCE_AMENDMENT_SEAL_RELATIVE),
    ):
        path = _validate_binding(root_path, governance.get(key), label=key)
        if path != _inside(root_path, relative):
            raise ProbabilityMetricErratumError(
                "probability metric erratum seal names noncanonical governance"
            )
    commit = str(seal.get("erratum_document_commit", ""))
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ProbabilityMetricErratumError(
            "probability metric erratum document commit is malformed"
        )
    if has_local_git:
        _validate_document_git_lineage(
            root=root_path,
            erratum_document_commit=commit,
            tip="HEAD",
            erratum_file=erratum_file,
        )
        _validate_probability_metric_erratum_seal_git_lineage(
            root=root_path,
            erratum_document_commit=commit,
            tip="HEAD",
            expected_sha256=_sha256_file(seal_file),
        )
    elif not allow_gitless_archive:
        raise ProbabilityMetricErratumError(
            "probability metric erratum lineage requires Git outside release replay"
        )
    return dict(seal)
