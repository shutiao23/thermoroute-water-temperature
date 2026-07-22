"""Git-backed pre-label chronology evidence for Route A.

The chronology receipt is intentionally narrower than a public
preregistration or an external timestamp.  It proves, for an honest repository
owner who does not rewrite history, that executable model bytes were committed
before candidate metadata and retrospective 2021--2023 predictors.  Every
scientific byte named by the receipt is replayed with ``git show`` and SHA-256;
the working-tree copy must be byte-identical as well.

This module uses only the Python standard library so that checking chronology
does not load a model, a dataframe, or any confirmation-period input.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable, Mapping


CHRONOLOGY_FORMAT = "thermoroute.route-a-prelabel-chronology.v1"
CHRONOLOGY_STATUS = "PASS_REPOSITORY_INTERNAL_PRELABEL_ORDER"
MODEL_SUITE_FORMAT = "thermoroute.route-a-model-suite.v1"
REPLAY_FORMAT = "thermoroute.route-a-development-replay.v1"
INPUT_MANIFEST_FORMAT = "thermoroute.route-a-prelabel-inputs.v1"
PROTOCOL_SEAL_FORMAT = "thermoroute.route-a-protocol-seal.v1"

DEFAULT_RECEIPT = "outputs/prelabel/route_a_prelabel_chronology_v1.json"
DEFAULT_PROTOCOL_SEAL = "protocols/route_a_protocol_seal_v1.json"
DEFAULT_MODEL_SUITE = "data_usgs/confirmatory_model_suite_v1.json"
DEFAULT_DEVELOPMENT_REPLAY = (
    "outputs/model_replay/route_a_development_replay_v1.json"
)
DEFAULT_CANDIDATE_TABLE = "data_usgs/confirmatory_candidate_sites_v1.csv"
DEFAULT_CANDIDATE_PROVENANCE = (
    "data_usgs/confirmatory_candidate_sites_v1.provenance.json"
)
DEFAULT_CANDIDATE_SNAPSHOT_INDEX = (
    "data_usgs/raw_snapshots/confirmatory-candidates-v1/snapshot_index.json"
)
DEFAULT_EXTERNAL_REGISTRY = "data_usgs/confirmatory_site_registry_v1.csv"
DEFAULT_EXTERNAL_LOCK = "data_usgs/confirmatory_site_registry_v1.lock.json"
DEFAULT_INPUT_MANIFEST = "data_usgs/confirmatory_actual_inputs_v1.json"

REQUIRED_GATE_PATHS = (
    "src/thermoroute/chronology.py",
    "src/thermoroute/outcome_qc.py",
    "scripts/28_freeze_prelabel_chronology.py",
    "tests/test_chronology.py",
    "protocols/route_a_outcome_qc_policy_v1.json",
)

STAGE09_RECEIPT_PATH = "outputs/models/route_a_stage09_completion.json"
STAGE09_ARTIFACT_PATHS = {
    "predictions": "outputs/predictions/usgs_predictions_stage9_v2.parquet",
    "prediction_sidecar": (
        "outputs/predictions/usgs_predictions_stage9_v2.parquet.meta.json"
    ),
    "scores": "outputs/tables/usgs_scores.csv",
    "report": "outputs/reports/usgs_experiment.md",
    "lightgbm_selection": (
        "outputs/tables/lightgbm_joint_validation_selection.csv"
    ),
    "thermoroute_pointer": "outputs/models/thermoroute_usgs_bundle.json",
    "lightgbm_pointer": "outputs/models/lightgbm_usgs_bundle.json",
    "components_pointer": "outputs/models/route_a_stage9_components.json",
}
STAGE09_ARTIFACT_LABELS = ("run_manifest", *STAGE09_ARTIFACT_PATHS)

STAGE09B_RECEIPT_PATH = "outputs/models/route_a_stage09b_completion.json"
STAGE09B_ARTIFACT_LABELS = (
    "run_manifest",
    "frozen_panel_spec",
    "panel",
    "registry",
    "predictor_bridge",
    "predictions",
    "prediction_sidecar",
    "architecture_budget",
    "architecture_budget_sidecar",
    "metric_summary",
    "metric_summary_sidecar",
    "report",
    "report_sidecar",
    "semantic_audit",
    "semantic_audit_sidecar",
)
STAGE09B_DATA_PATHS = {
    "frozen_panel_spec": "data_usgs/frozen_panel_v1.json",
    "panel": "data_usgs/panel_usgs_120v2.parquet",
    "registry": "data_usgs/station_registry_v1.csv",
    "predictor_bridge": "data_usgs/development_predictor_bridge_v1.json",
}
STAGE09B_ARM_SEEDS = (
    ("PlainMLP-7var", (0, 1, 2, 3, 4)),
    ("PlainCausalTCN-7var", (0, 1, 2, 3, 4)),
    ("ThermoRoute-ladder-01_WTEMP", (0, 1, 2)),
    ("ThermoRoute-ladder-02_plus_FLOW", (0, 1, 2)),
    ("ThermoRoute-ladder-03_plus_TEMP", (0, 1, 2)),
    ("ThermoRoute-ladder-04_plus_PRCP", (0, 1, 2)),
    ("ThermoRoute-ladder-05_plus_RHMEAN", (0, 1, 2)),
    ("ThermoRoute-ladder-06_plus_DH", (0, 1, 2)),
    ("ThermoRoute-ladder-07_plus_WDSP", (0, 1, 2)),
)
STAGE09B_MEMBERS = tuple(
    (arm_id, seed)
    for arm_id, seeds in STAGE09B_ARM_SEEDS
    for seed in seeds
)


def _stage09b_scientific_comparison_registry() -> list[dict[str, Any]]:
    full = "ThermoRoute-ladder-07_plus_WDSP"
    controls = [
        {
            "comparison_family": "full_vs_control",
            "comparison_id": f"{full}-minus-{reference}",
            "candidate_arm_id": full,
            "reference_arm_id": reference,
            "seeds": [0, 1, 2],
        }
        for reference in ("PlainMLP-7var", "PlainCausalTCN-7var")
    ]
    ladder_arms = [arm_id for arm_id, _seeds in STAGE09B_ARM_SEEDS[2:]]
    adjacent = [
        {
            "comparison_family": "adjacent_feature_ladder",
            "comparison_id": f"{candidate}-minus-{reference}",
            "candidate_arm_id": candidate,
            "reference_arm_id": reference,
            "seeds": [0, 1, 2],
        }
        for reference, candidate in zip(
            ladder_arms[:-1], ladder_arms[1:], strict=True,
        )
    ]
    return controls + adjacent


def _validate_stage09b_scientific_summary(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "format", "metric_summary_format", "primary_member_estimand",
        "secondary_member_estimands", "paired_descriptive_effects",
    }:
        raise ChronologyError("Stage-09b scientific summary schema changed")
    primary = {
        "name": "median_across_stations_of_within_station_rmse_c",
        "column": "median_station_rmse_c",
        "unit": "degree_Celsius",
        "aggregation": "median_of_within_station_RMSE",
        "station_weighting": "one_station_one_value",
    }
    secondary = {
        "micro_rmse_c": {
            "role": "secondary_not_primary_estimand",
            "aggregation": "RMSE_over_all_forecast_keys",
        },
        "micro_mae_c": {
            "role": "secondary_not_primary_estimand",
            "aggregation": "MAE_over_all_forecast_keys",
        },
    }
    paired = value.get("paired_descriptive_effects")
    comparisons = _stage09b_scientific_comparison_registry()
    ladder_variables = (
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
    if (
        value.get("format")
        != "thermoroute.development-controls-scientific-summary.v1"
        or value.get("metric_summary_format")
        != "thermoroute.development-controls-metric-summary.v2"
        or value.get("primary_member_estimand") != primary
        or value.get("secondary_member_estimands") != secondary
        or not isinstance(paired, Mapping)
        or set(paired) != {
            "estimand", "effect_convention", "negative_favours", "same_seed",
            "exact_common_forecast_keys_verified", "comparison_registry",
            "feature_ladder_order", "feature_ladder_fixed_order_path_dependent",
            "independent_feature_contribution_claimed", "causal_effect_claimed",
            "records_sha256", "records",
        }
        or paired.get("estimand")
        != "median_across_stations_of_candidate_rmse_minus_reference_rmse_c"
        or paired.get("effect_convention") != "candidate_minus_reference"
        or paired.get("negative_favours") != "candidate"
        or paired.get("same_seed") is not True
        or paired.get("exact_common_forecast_keys_verified") is not True
        or paired.get("comparison_registry") != comparisons
        or paired.get("feature_ladder_order") != [
            {"rung": rung, "variables": variables}
            for rung, variables in ladder_variables
        ]
        or paired.get("feature_ladder_fixed_order_path_dependent") is not True
        or paired.get("independent_feature_contribution_claimed") is not False
        or paired.get("causal_effect_claimed") is not False
    ):
        raise ChronologyError("Stage-09b scientific estimand contract changed")
    records = paired.get("records")
    expected_identities = [
        (
            comparison["comparison_family"], comparison["comparison_id"],
            comparison["candidate_arm_id"], comparison["reference_arm_id"],
            seed, split, horizon,
        )
        for comparison in comparisons
        for seed in (0, 1, 2)
        for split in ("calib", "test", "val")
        for horizon in (1, 3, 7)
    ]
    if not isinstance(records, list) or len(records) != len(expected_identities):
        raise ChronologyError("Stage-09b paired-effect registry is incomplete")
    observed_identities: list[tuple[object, ...]] = []
    record_keys = {
        "comparison_family", "comparison_id", "candidate_arm_id",
        "reference_arm_id", "seed", "split", "horizon", "common_forecast_keys",
        "stations", "median_paired_station_rmse_difference_c",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != record_keys:
            raise ChronologyError("Stage-09b paired-effect record schema changed")
        effect = record.get("median_paired_station_rmse_difference_c")
        if (
            type(record.get("seed")) is not int
            or type(record.get("horizon")) is not int
            or type(record.get("common_forecast_keys")) is not int
            or int(record["common_forecast_keys"]) < 1
            or type(record.get("stations")) is not int
            or int(record["stations"]) < 1
            or type(effect) not in {int, float}
        ):
            raise ChronologyError("Stage-09b paired-effect record is malformed")
        assert isinstance(effect, (int, float))
        if not math.isfinite(float(effect)):
            raise ChronologyError("Stage-09b paired-effect record is malformed")
        observed_identities.append((
            record["comparison_family"], record["comparison_id"],
            record["candidate_arm_id"], record["reference_arm_id"],
            record["seed"], record["split"], record["horizon"],
        ))
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if (
        observed_identities != expected_identities
        or paired.get("records_sha256") != hashlib.sha256(encoded).hexdigest()
    ):
        raise ChronologyError("Stage-09b paired-effect evidence changed")

PROTECTED_DIRECTORIES = ("src", "scripts", "tests", "protocols", ".github")
PROTECTED_EXACT_FILES = (".gitignore", "pyproject.toml")
PROTECTED_ROOT_PATTERNS = (
    "requirements*.txt",
    "*lock*",
    "*.lock",
)
SOURCE_INVENTORY_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/**/*.py",
    "protocols/**/*.json",
    "protocols/**/*.md",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock*.txt",
)

# These names are fixed independently of the paths discovered from the input
# manifest.  Directory entries are checked recursively with ``git ls-tree``.
FIXED_PRELABEL_ABSENCE_PATHS = (
    DEFAULT_RECEIPT,
    DEFAULT_CANDIDATE_TABLE,
    DEFAULT_CANDIDATE_PROVENANCE,
    "data_usgs/raw_snapshots/confirmatory-candidates-v1",
    DEFAULT_EXTERNAL_REGISTRY,
    DEFAULT_EXTERNAL_LOCK,
    DEFAULT_INPUT_MANIFEST,
    "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1",
    "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1",
    "data_usgs/confirmatory_predictors",
    "data_usgs/confirmatory_opening_authorization_v1.json",
    "data_usgs/confirmatory",
    "data_usgs/confirmatory_outcomes",
    "outputs/confirmatory",
)

EVIDENCE_SCOPE = (
    "repository-internal Git ancestry and SHA-256 evidence for an honest owner; "
    "not proof against owner-controlled Git-history rewriting"
)

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


class ChronologyError(RuntimeError):
    """A Route-A chronology assertion cannot be established exactly."""


def _safe_git_environment() -> dict[str, str]:
    """Return a deterministic Git environment with repository overrides removed."""
    forbidden = sorted(
        name
        for name in os.environ
        if name in _FORBIDDEN_AMBIENT_GIT_VARIABLES
        or name.startswith("GIT_CONFIG_KEY_")
        or name.startswith("GIT_CONFIG_VALUE_")
        or (name == "GIT_NO_REPLACE_OBJECTS" and os.environ[name] != "1")
    )
    if forbidden:
        raise ChronologyError(
            f"ambient Git repository/configuration override is prohibited: {forbidden}"
        )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("GIT_")
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


def _git_command(root: Path, *arguments: str) -> list[str]:
    return [
        "git",
        "--no-replace-objects",
        "-c",
        "core.useReplaceRefs=false",
        "-C",
        str(root),
        *arguments,
    ]


def assert_no_hidden_index_flags(root: str | Path) -> None:
    """Reject index flags that can hide working-byte changes from Git status."""
    root = _require_git_root(root)
    records = _git(root, "ls-files", "-v", "-z").split(b"\0")
    hidden: list[str] = []
    for record in records:
        if not record:
            continue
        try:
            tag = chr(record[0])
            relative = record[2:].decode("utf-8", errors="strict")
        except (UnicodeDecodeError, IndexError) as exc:
            raise ChronologyError("cannot parse Git index flags") from exc
        if tag == "S" or tag.islower():
            hidden.append(f"{tag} {relative}")
    if hidden:
        raise ChronologyError(
            "Git assume-unchanged/skip-worktree flags are prohibited: "
            f"{hidden[:10]}"
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _sha256_json(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _repro_sha256_json(value: Mapping[str, Any]) -> str:
    """Match :func:`thermoroute.repro.sha256_json` without importing runtime deps."""
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _portable_current_binding(
    root: Path,
    value: object,
    *,
    label: str,
) -> str:
    """Replay one Git-derived binding against extracted archive bytes.

    A release archive deliberately has no mutable ``.git`` directory.  Its
    outer verifier replays commit ancestry from the included Git bundle; this
    narrower check makes the isolated scorer independently require that every
    chronology-bound current artifact still has the frozen digest and byte
    count.  It must never be presented as a replacement for the Git replay.
    """
    if not isinstance(value, Mapping):
        raise ChronologyError(f"{label} binding is not an object")
    if set(value) != {"path", "sha256", "byte_count", "git_blob_oid"}:
        raise ChronologyError(f"{label} binding schema changed")
    path = _normalise_path(str(value.get("path", "")))
    candidate = (root / path).resolve()
    if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
        raise ChronologyError(f"{label} archive artifact is absent or unsafe: {path}")
    digest, byte_count = _sha256_file(candidate)
    if digest != value.get("sha256") or byte_count != value.get("byte_count"):
        raise ChronologyError(f"{label} archive bytes differ from chronology: {path}")
    oid = str(value.get("git_blob_oid", ""))
    if not oid or any(character not in "0123456789abcdef" for character in oid):
        raise ChronologyError(f"{label} Git blob OID is malformed: {path}")
    return path


def _validate_portable_receipt_bytes(
    root: Path,
    document: Mapping[str, Any],
) -> None:
    """Validate current bytes in a gitless release without overstating order.

    Commit ancestry is verified separately by ``scripts/verify_release.py``
    from the mandatory Git bundle.  This function checks only the receipt's
    immutable structure and the extracted bytes consumed by the scorer.
    """
    if document.get("status") != CHRONOLOGY_STATUS:
        raise ChronologyError("chronology receipt status changed")
    if (
        document.get("external_timestamp_or_public_preregistration") is not False
        or document.get("independent_custodian_or_worm_storage") is not False
        or document.get("evidence_scope") != EVIDENCE_SCOPE
    ):
        raise ChronologyError("chronology evidence-scope disclosure changed")
    if document.get("post_freeze_artifact_mutation_count") != 0:
        raise ChronologyError("chronology reports a post-freeze artifact mutation")
    order = document.get("order")
    if not isinstance(order, Mapping) or order.get("strict_order_verified") is not True:
        raise ChronologyError("chronology receipt lacks its strict-order assertion")
    commits = (
        order.get("model_freeze_commit"),
        order.get("input_evidence_commit"),
        order.get("receipt_creation_base_commit"),
    )
    if any(
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        for commit in commits
    ) or len(set(commits)) != 3:
        raise ChronologyError("chronology commit identities are malformed or duplicated")

    observed_by_field: dict[str, set[str]] = {}
    bindings_by_path: dict[str, Mapping[str, Any]] = {}
    for field, minimum in (
        ("required_gate_files_at_model_freeze", len(REQUIRED_GATE_PATHS)),
        ("model_source_control_artifacts", 1),
        ("model_freeze_artifacts", 1),
        ("input_evidence_artifacts", 1),
    ):
        values = document.get(field)
        if not isinstance(values, list) or len(values) < minimum:
            raise ChronologyError(f"chronology {field} registry is incomplete")
        seen: set[str] = set()
        for index, value in enumerate(values):
            path = _portable_current_binding(
                root, value, label=f"chronology {field}[{index}]"
            )
            if path in seen:
                raise ChronologyError(f"chronology binds an artifact twice: {path}")
            seen.add(path)
            assert isinstance(value, Mapping)
            bindings_by_path[path] = value
        observed_by_field[field] = seen

    declared_control = observed_by_field["model_source_control_artifacts"]
    if declared_control != _working_model_control_paths(root):
        raise ChronologyError(
            "gitless archive source/control path set differs from chronology"
        )
    source_paths: set[str] = set()
    for pattern in SOURCE_INVENTORY_PATTERNS:
        for source_path in root.glob(pattern):
            if source_path.is_file() and "__pycache__" not in source_path.parts:
                source_paths.add(source_path.relative_to(root).as_posix())
    inventory = {
        path: str(bindings_by_path[path]["sha256"])
        for path in sorted(source_paths)
        if path in bindings_by_path
    }
    if (
        set(inventory) != source_paths
        or _repro_sha256_json(inventory) != document.get("source_tree_sha256")
    ):
        raise ChronologyError("gitless archive source-tree lineage changed")

    protocol = document.get("protocol_history")
    if not isinstance(protocol, Mapping):
        raise ChronologyError("chronology protocol history is absent")
    _portable_current_binding(
        root, protocol.get("seal"), label="chronology protocol seal"
    )
    declared = protocol.get("declared_git_show_bindings")
    if not isinstance(declared, list) or {item.get("role") for item in declared
                                        if isinstance(item, Mapping)} != {
        "original_markdown", "final_json", "final_markdown"
    }:
        raise ChronologyError("chronology protocol Git-show registry changed")
    # Current bytes must equal the final protocol bindings.  The original
    # Markdown intentionally has the same path but older bytes and is replayed
    # only from the Git bundle by the outer release verifier.
    for item in declared:
        assert isinstance(item, Mapping)
        if item.get("role") == "original_markdown":
            continue
        path = _normalise_path(str(item.get("path", "")))
        candidate = (root / path).resolve()
        if root not in candidate.parents or candidate.is_symlink() or not candidate.is_file():
            raise ChronologyError(f"final protocol artifact is absent: {path}")
        digest, _ = _sha256_file(candidate)
        if digest != item.get("sha256"):
            raise ChronologyError(f"final protocol artifact differs: {path}")


def _git(root: Path, *arguments: str, check: bool = True) -> bytes:
    try:
        result = subprocess.run(
            _git_command(root, *arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
            check=False,
        )
    except OSError as exc:  # pragma: no cover - Git is available in CI
        raise ChronologyError("Git is required for Route-A chronology") from exc
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ChronologyError(
            f"Git command failed ({' '.join(arguments)}): {detail or result.returncode}"
        )
    return result.stdout


def _require_git_root(root: str | Path) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ChronologyError(f"repository root is absent: {root}")
    output = _git(root, "rev-parse", "--show-toplevel").decode().strip()
    if Path(output).resolve() != root:
        raise ChronologyError(
            f"chronology root must be the Git top-level directory: {output}"
        )
    replacements = _git(
        root, "for-each-ref", "--format=%(refname)", "refs/replace"
    ).decode().splitlines()
    if replacements:
        raise ChronologyError(
            f"Git replace refs are prohibited for pre-label chronology: {replacements}"
        )
    if _git(root, "rev-parse", "--is-shallow-repository").decode().strip() != "false":
        raise ChronologyError("a shallow Git repository cannot establish chronology")
    for label, relative in (
        ("legacy grafts", "info/grafts"),
        ("object alternates", "objects/info/alternates"),
    ):
        raw_path = _git(root, "rev-parse", "--git-path", relative).decode().strip()
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            payload = candidate.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ChronologyError(f"cannot audit Git {label}") from exc
        if payload.strip():
            raise ChronologyError(f"Git {label} are prohibited for chronology")
    return root


def _resolve_commit(root: Path, revision: str) -> str:
    raw = str(revision).strip()
    if not raw or raw.startswith("-"):
        raise ChronologyError("empty or option-like Git revision")
    output = _git(root, "rev-parse", "--verify", f"{raw}^{{commit}}").decode().strip()
    if len(output) != 40 or any(character not in "0123456789abcdef" for character in output):
        raise ChronologyError(f"revision did not resolve to a full SHA-1 commit: {revision}")
    return output


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        _git_command(root, "merge-base", "--is-ancestor", ancestor, descendant),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_safe_git_environment(),
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise ChronologyError("cannot evaluate Git ancestry")
    return result.returncode == 0


def _strictly_precedes(root: Path, earlier: str, later: str, *, label: str) -> None:
    if earlier == later or not _is_ancestor(root, earlier, later):
        raise ChronologyError(f"required strict Git order is false: {label}")


def _normalise_path(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or raw != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\x00" in raw
    ):
        raise ChronologyError(f"artifact path is not canonical repository-relative: {raw!r}")
    return raw


def _relative(root: Path, path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return _normalise_path(value.as_posix())
    try:
        return _normalise_path(value.resolve().relative_to(root).as_posix())
    except ValueError as exc:
        raise ChronologyError(f"artifact escapes repository: {value}") from exc


def _join_relative(parent_file: str, child: object) -> str:
    raw = str(child).replace("\\", "/")
    if PurePosixPath(raw).is_absolute():
        raise ChronologyError(f"nested artifact path is absolute: {raw}")
    joined = (PurePosixPath(parent_file).parent / PurePosixPath(raw)).as_posix()
    return _normalise_path(joined)


def _git_show(root: Path, commit: str, path: str) -> bytes:
    path = _normalise_path(path)
    result = subprocess.run(
        _git_command(root, "show", f"{commit}:{path}"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_safe_git_environment(),
        check=False,
    )
    if result.returncode:
        raise ChronologyError(f"required Git blob is absent at {commit[:12]}: {path}")
    return result.stdout


def _git_path_exists(root: Path, commit: str, path: str) -> bool:
    result = subprocess.run(
        _git_command(
            root, "cat-file", "-e", f"{commit}:{_normalise_path(path)}"
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_safe_git_environment(),
        check=False,
    )
    return result.returncode == 0


def _git_tree_paths(root: Path, commit: str, path: str) -> list[str]:
    output = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        commit,
        "--",
        _normalise_path(path),
    )
    return sorted(
        item.decode("utf-8")
        for item in output.split(b"\0")
        if item
    )


def _json_from_git(root: Path, commit: str, path: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_git_show(root, commit, path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChronologyError(f"{label} is not valid UTF-8 JSON at {commit[:12]}") from exc
    if not isinstance(value, dict):
        raise ChronologyError(f"{label} is not a JSON object")
    return value


def _binding(
    root: Path,
    commit: str,
    path: str,
    *,
    expected_sha256: object | None = None,
) -> dict[str, Any]:
    path = _normalise_path(path)
    oid = _git(root, "rev-parse", f"{commit}:{path}").decode().strip()
    if _git(root, "cat-file", "-t", oid).decode().strip() != "blob":
        raise ChronologyError(f"Git object is not a blob: {path}")
    try:
        process = subprocess.Popen(
            _git_command(root, "cat-file", "blob", oid),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_safe_git_environment(),
        )
    except OSError as exc:  # pragma: no cover - Git is available in CI
        raise ChronologyError("cannot stream Git artifact bytes") from exc
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        process.kill()
        raise ChronologyError("cannot open Git artifact stream")
    git_digest = hashlib.sha256()
    git_byte_count = 0
    while chunk := process.stdout.read(1024 * 1024):
        git_digest.update(chunk)
        git_byte_count += len(chunk)
    error = process.stderr.read().decode("utf-8", errors="replace").strip()
    returncode = process.wait()
    if returncode:
        raise ChronologyError(f"cannot stream Git blob for {path}: {error}")
    digest = git_digest.hexdigest()
    if expected_sha256 is not None and digest != str(expected_sha256):
        raise ChronologyError(
            f"declared SHA-256 differs from Git blob at {commit[:12]}: {path}"
        )
    current = root / path
    if current.is_symlink() or not current.is_file():
        raise ChronologyError(f"current artifact is absent or a symlink: {path}")
    current_digest, current_byte_count = _sha256_file(current)
    if current_digest != digest or current_byte_count != git_byte_count:
        raise ChronologyError(f"working-tree bytes differ from {commit[:12]}: {path}")
    return {
        "path": path,
        "sha256": digest,
        "byte_count": git_byte_count,
        "git_blob_oid": oid,
    }


def _add_binding(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    path: str,
    *,
    expected_sha256: object | None = None,
) -> None:
    path = _normalise_path(path)
    value = _binding(root, commit, path, expected_sha256=expected_sha256)
    existing = output.get(path)
    if existing is not None and existing != value:
        raise ChronologyError(f"artifact has conflicting bindings: {path}")
    output[path] = value


def _declared_root_binding(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, Mapping) or not {"path", "sha256"} <= set(value):
        raise ChronologyError(f"{label} lacks a path/SHA-256 binding")
    path = _normalise_path(str(value["path"]))
    _add_binding(output, root, commit, path, expected_sha256=value["sha256"])
    return path


def _collect_prediction_binding(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    value: object,
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ChronologyError(f"{label} lacks development-prediction lineage")
    artifact = value.get("artifact")
    path = _declared_root_binding(
        output, root, commit, artifact, label=f"{label} prediction"
    )
    assert isinstance(artifact, Mapping)  # narrowed by _declared_root_binding
    sidecar = artifact.get("sidecar")
    if sidecar is None:
        raise ChronologyError(f"{label} prediction lacks its sidecar binding")
    _declared_root_binding(
        output, root, commit, sidecar, label=f"{label} prediction sidecar"
    )
    if path == "":  # pragma: no cover - canonical paths cannot be empty
        raise ChronologyError("empty prediction path")


def _collect_lightgbm_bundle(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    manifest_path: str,
) -> None:
    document = _json_from_git(
        root, commit, manifest_path, label="LightGBM bundle manifest"
    )
    if document.get("format") != "thermoroute.lightgbm-bundle.v1":
        raise ChronologyError("unsupported LightGBM bundle in model suite")
    models = document.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ChronologyError("LightGBM bundle has no model registry")
    model_count = 0
    for horizons in models.values():
        if not isinstance(horizons, Mapping):
            raise ChronologyError("LightGBM horizon registry is malformed")
        for heads in horizons.values():
            if not isinstance(heads, Mapping):
                raise ChronologyError("LightGBM head registry is malformed")
            for item in heads.values():
                if not isinstance(item, Mapping) or not {"path", "sha256"} <= set(item):
                    raise ChronologyError("LightGBM model binding is malformed")
                model_path = _join_relative(manifest_path, item["path"])
                _add_binding(
                    output,
                    root,
                    commit,
                    model_path,
                    expected_sha256=item["sha256"],
                )
                model_count += 1
    if model_count < 1:
        raise ChronologyError("LightGBM bundle resolved no model files")
    _collect_prediction_binding(
        output,
        root,
        commit,
        document.get("development_prediction"),
        label="LightGBM",
    )


def _collect_development_bridge(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    development: Mapping[str, Any],
) -> None:
    bridge_path = _declared_root_binding(
        output, root, commit, development.get("predictor_bridge"),
        label="model-suite development predictor bridge",
    )
    bridge = _json_from_git(
        root, commit, bridge_path, label="development predictor bridge",
    )
    bridge_source = bridge.get("source_tree_sha256")
    if (
        bridge.get("format") != "thermoroute.development-predictor-bridge.v1"
        or bridge.get("status") != "PASS_EXACT_PRODUCT_BRIDGE"
        or bridge.get("outcome_values_requested_or_read") is not False
        or not isinstance(bridge_source, str)
        or len(bridge_source) != 64
        or any(character not in "0123456789abcdef" for character in bridge_source)
        or bridge.get("panel") != development.get("panel")
        or bridge.get("registry") != development.get("registry")
    ):
        raise ChronologyError("development predictor bridge is stale or malformed")
    normalized = bridge.get("normalized")
    indexes = bridge.get("raw_snapshot_indexes")
    if (
        not isinstance(normalized, Mapping)
        or set(normalized) != {"frozen", "refreshed"}
        or not isinstance(indexes, Mapping)
        or set(indexes) != {"daymet", "gridmet", "gridmet_schema"}
    ):
        raise ChronologyError("development predictor bridge dependency registry changed")
    for name, binding in normalized.items():
        _declared_root_binding(
            output, root, commit, binding,
            label=f"development predictor bridge normalized/{name}",
        )
    for name, binding in indexes.items():
        index_path = _declared_root_binding(
            output, root, commit, binding,
            label=f"development predictor bridge raw/{name}",
        )
        if PurePosixPath(index_path).name != "snapshot_index_v2.json":
            raise ChronologyError(
                "development predictor bridge lacks metadata-byte-bound index v2"
            )
        _collect_snapshot_files(
            output, root, commit, index_path, require_metadata_binding=True,
        )
    for name in ("report", "request_map"):
        _declared_root_binding(
            output, root, commit, bridge.get(name),
            label=f"development predictor bridge {name}",
        )


def _collect_preopening_receipts(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    suite: Mapping[str, Any],
) -> None:
    gates = suite.get("preopening_gates")
    if not isinstance(gates, Mapping) or set(gates) != {
        "stage09_completion", "stage09b_development_controls",
    }:
        raise ChronologyError("model suite lacks exact Stage-09/09b completion gates")
    for gate_name, expected_path, expected_format, expected_status in (
        (
            "stage09_completion",
            STAGE09_RECEIPT_PATH,
            "thermoroute.stage09-completion-receipt.v1",
            "PASS_FORMAL_STAGE09_COMPLETE",
        ),
        (
            "stage09b_development_controls",
            STAGE09B_RECEIPT_PATH,
            "thermoroute.stage09b-completion-receipt.v3",
            "PASS_STAGE09B_BEST_MODEL_STATE_PREDICTION_REPLAY",
        ),
    ):
        receipt_path = _declared_root_binding(
            output, root, commit, gates.get(gate_name), label=gate_name,
        )
        if receipt_path != expected_path:
            raise ChronologyError(f"{gate_name} receipt path is noncanonical")
        receipt = _json_from_git(root, commit, receipt_path, label=gate_name)
        unhashed = dict(receipt)
        receipt_self = unhashed.pop("receipt_self_sha256", None)
        if (
            receipt.get("format") != expected_format
            or receipt.get("status") != expected_status
            or receipt_self != _repro_sha256_json(unhashed)
        ):
            raise ChronologyError(f"{gate_name} receipt is stale or malformed")
        if gate_name == "stage09_completion":
            expected_receipt_keys = {
                "format", "status", "stage", "run_id", "run_identity",
                "formal_configuration", "confirmation_outcomes_requested_or_read",
                "artifacts", "receipt_self_sha256",
            }
            if (
                set(receipt) != expected_receipt_keys
                or receipt.get("stage") != "09_usgs_experiment"
                or receipt.get("confirmation_outcomes_requested_or_read") is not False
                or not isinstance(receipt.get("run_id"), str)
                or not receipt["run_id"]
                or not isinstance(receipt.get("run_identity"), Mapping)
                or not isinstance(receipt.get("formal_configuration"), Mapping)
            ):
                raise ChronologyError("Stage-09 receipt contract changed")
            expected_artifact_labels = set(STAGE09_ARTIFACT_LABELS)
        else:
            expected_receipt_keys = {
                "format", "status", "stage", "run_id", "run_identity",
                "formal_configuration", "evidence_scope",
                "training_replay_verified", "matrix_audit", "member_registry",
                "best_model_state_prediction_replay_verified",
                "artifacts", "post_2020_outcomes_requested_or_read",
                "receipt_self_sha256",
            }
            if (
                set(receipt) != expected_receipt_keys
                or receipt.get("stage") != "09b_development_controls"
                or receipt.get("evidence_scope")
                != "best_model_state_prediction_replay"
                or receipt.get("training_replay_verified") is not False
                or receipt.get("best_model_state_prediction_replay_verified") is not True
                or receipt.get("post_2020_outcomes_requested_or_read") is not False
                or not isinstance(receipt.get("run_id"), str)
                or not receipt["run_id"]
                or not isinstance(receipt.get("run_identity"), Mapping)
                or not isinstance(receipt.get("formal_configuration"), Mapping)
                or not isinstance(receipt.get("matrix_audit"), Mapping)
            ):
                raise ChronologyError("Stage-09b receipt contract changed")
            expected_artifact_labels = set(STAGE09B_ARTIFACT_LABELS)
        artifacts = receipt.get("artifacts")
        if (
            not isinstance(artifacts, Mapping)
            or set(artifacts) != expected_artifact_labels
        ):
            raise ChronologyError(f"{gate_name} artifact registry changed")
        resolved: dict[str, str] = {}
        for label, binding in artifacts.items():
            if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                raise ChronologyError(f"{gate_name} {label} binding is not exact")
            resolved[str(label)] = _declared_root_binding(
                output, root, commit, binding,
                label=f"{gate_name} {label}",
            )
        run_id = str(receipt["run_id"])
        if gate_name == "stage09_completion":
            expected_paths = {
                "run_manifest": (
                    f"outputs/runs/09_usgs_experiment/{run_id}/run.json"
                ),
                **STAGE09_ARTIFACT_PATHS,
            }
            if resolved != expected_paths:
                raise ChronologyError("Stage-09 artifact paths are noncanonical")
            continue

        run_dir = f"outputs/runs/09b_development_controls/{run_id}"
        expected_paths = {
            "run_manifest": f"{run_dir}/run.json",
            **STAGE09B_DATA_PATHS,
            "predictions": f"{run_dir}/development_controls_predictions.parquet",
            "prediction_sidecar": (
                f"{run_dir}/development_controls_predictions.parquet.meta.json"
            ),
            "architecture_budget": (
                f"{run_dir}/development_controls_architecture_budget.csv"
            ),
            "architecture_budget_sidecar": (
                f"{run_dir}/development_controls_architecture_budget.csv.meta.json"
            ),
            "metric_summary": f"{run_dir}/development_controls_metric_summary.csv",
            "metric_summary_sidecar": (
                f"{run_dir}/development_controls_metric_summary.csv.meta.json"
            ),
            "report": f"{run_dir}/development_controls_report.md",
            "report_sidecar": f"{run_dir}/development_controls_report.md.meta.json",
            "semantic_audit": f"{run_dir}/development_controls_semantic_audit.json",
            "semantic_audit_sidecar": (
                f"{run_dir}/development_controls_semantic_audit.json.meta.json"
            ),
        }
        if resolved != expected_paths:
            raise ChronologyError("Stage-09b artifact paths are noncanonical")
        members = receipt.get("member_registry")
        if not isinstance(members, list) or len(members) != len(STAGE09B_MEMBERS):
            raise ChronologyError("Stage-09b receipt does not bind 31 exact members")
        member_paths: dict[tuple[str, int], tuple[str, str, str, str]] = {}
        for member, expected_member in zip(members, STAGE09B_MEMBERS):
            if (
                not isinstance(member, Mapping)
                or set(member) != {
                    "arm_id", "seed", "checkpoint", "checkpoint_sidecar",
                    "prediction", "prediction_sidecar"
                }
                or (member.get("arm_id"), member.get("seed")) != expected_member
            ):
                raise ChronologyError("Stage-09b member binding is malformed")
            arm_id, seed = expected_member
            expected_prediction = (
                f"{run_dir}/arm_predictions/{arm_id}/seed{seed}.parquet"
            )
            expected_sidecar = f"{expected_prediction}.meta.json"
            expected_checkpoint = f"{run_dir}/checkpoints/{arm_id}/seed{seed}.pt"
            expected_checkpoint_sidecar = f"{expected_checkpoint}.meta.json"
            observed_paths = []
            for label in (
                "prediction", "prediction_sidecar", "checkpoint", "checkpoint_sidecar"
            ):
                binding = member.get(label)
                if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
                    raise ChronologyError("Stage-09b member binding is not exact")
                observed_paths.append(_declared_root_binding(
                    output, root, commit, member.get(label),
                    label=f"Stage-09b member {label}",
                ))
            if observed_paths != [
                expected_prediction, expected_sidecar,
                expected_checkpoint, expected_checkpoint_sidecar,
            ]:
                raise ChronologyError("Stage-09b member path is noncanonical")
            member_paths[expected_member] = (
                expected_prediction, expected_sidecar,
                expected_checkpoint, expected_checkpoint_sidecar,
            )
        semantic_path = resolved.get("semantic_audit")
        assert semantic_path is not None
        semantic = _json_from_git(
            root, commit, semantic_path, label="Stage-09b semantic audit",
        )
        stable_semantic = dict(semantic)
        semantic_self = stable_semantic.pop("semantic_audit_self_sha256", None)
        expected_semantic_keys = {
            "format", "status", "run_id", "evidence_scope",
            "training_replay_verified", "post_2020_outcomes_requested_or_read",
            "best_model_state_prediction_replay_verified",
            "matrix_audit", "canonical_window_registry", "scientific_summary",
            "members",
            "derived_artifacts", "semantic_audit_self_sha256",
        }
        if (
            set(semantic) != expected_semantic_keys
            or semantic.get("format")
            != "thermoroute.development-controls-semantic-audit.v3"
            or semantic.get("status")
            != "PASS_BEST_MODEL_STATE_PREDICTION_REPLAY"
            or semantic.get("run_id") != run_id
            or semantic.get("evidence_scope") != "best_model_state_prediction_replay"
            or semantic.get("training_replay_verified") is not False
            or semantic.get("best_model_state_prediction_replay_verified") is not True
            or semantic.get("post_2020_outcomes_requested_or_read") is not False
            or semantic.get("matrix_audit") != receipt.get("matrix_audit")
            or semantic_self != _repro_sha256_json(stable_semantic)
        ):
            raise ChronologyError("Stage-09b semantic audit changed")
        _validate_stage09b_scientific_summary(semantic.get("scientific_summary"))
        registry = semantic.get("canonical_window_registry")
        if (
            not isinstance(registry, Mapping)
            or set(registry) != {
                "sha256", "common_forecast_keys", "train_examples_per_epoch",
                "train_registry_sha256",
            }
            or any(
                not isinstance(registry.get(key), str)
                or len(str(registry[key])) != 64
                or any(character not in "0123456789abcdef" for character in str(registry[key]))
                for key in ("sha256", "train_registry_sha256")
            )
            or type(registry.get("common_forecast_keys")) is not int
            or int(registry["common_forecast_keys"]) < 1
            or type(registry.get("train_examples_per_epoch")) is not int
            or int(registry["train_examples_per_epoch"]) < 1
        ):
            raise ChronologyError("Stage-09b canonical window registry changed")

        semantic_members = semantic.get("members")
        if not isinstance(semantic_members, list) or len(semantic_members) != len(
            STAGE09B_MEMBERS
        ):
            raise ChronologyError("Stage-09b semantic member registry changed")

        def descriptor(path: str) -> dict[str, Any]:
            binding = output[path]
            return {"sha256": binding["sha256"], "bytes": binding["byte_count"]}

        for row, expected_member in zip(semantic_members, STAGE09B_MEMBERS):
            prediction_path, sidecar_path, checkpoint_path, checkpoint_sidecar = (
                member_paths[expected_member]
            )
            digest = row.get("normalised_prediction_sha256") if isinstance(
                row, Mapping
            ) else None
            if (
                not isinstance(row, Mapping)
                or set(row) != {
                    "arm_id", "seed", "prediction", "prediction_sidecar",
                    "checkpoint", "checkpoint_sidecar",
                    "normalised_prediction_sha256",
                    "best_model_state_prediction_replay_verified",
                }
                or (row.get("arm_id"), row.get("seed")) != expected_member
                or row.get("prediction") != descriptor(prediction_path)
                or row.get("prediction_sidecar") != descriptor(sidecar_path)
                or row.get("checkpoint") != descriptor(checkpoint_path)
                or row.get("checkpoint_sidecar") != descriptor(checkpoint_sidecar)
                or row.get("best_model_state_prediction_replay_verified") is not True
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ChronologyError("Stage-09b semantic member evidence changed")
        semantic_derived = semantic.get("derived_artifacts")
        derived_labels = {
            "architecture_budget": ("architecture_budget", "architecture_budget_sidecar"),
            "combined_predictions": ("predictions", "prediction_sidecar"),
            "metric_summary": ("metric_summary", "metric_summary_sidecar"),
            "report": ("report", "report_sidecar"),
        }
        if not isinstance(semantic_derived, Mapping) or set(
            semantic_derived
        ) != set(derived_labels):
            raise ChronologyError("Stage-09b semantic derived-artifact registry changed")
        for label, (artifact_label, sidecar_label) in derived_labels.items():
            if semantic_derived[label] != {
                "artifact": descriptor(resolved[artifact_label]),
                "sidecar": descriptor(resolved[sidecar_label]),
            }:
                raise ChronologyError("Stage-09b semantic artifact evidence changed")


def _collect_model_artifacts(
    root: Path,
    commit: str,
    suite_path: str,
    replay_path: str,
) -> tuple[list[dict[str, Any]], str, str]:
    output: dict[str, dict[str, Any]] = {}
    _add_binding(output, root, commit, suite_path)
    suite = _json_from_git(root, commit, suite_path, label="model-suite registry")
    if (
        suite.get("format") != MODEL_SUITE_FORMAT
        or suite.get("status") != "FROZEN_BEFORE_LABEL_OPENING"
    ):
        raise ChronologyError("model-suite registry is not frozen Route A")

    development = suite.get("development_contract")
    if not isinstance(development, Mapping):
        raise ChronologyError("model suite lacks development contract")
    for name in ("frozen_panel_spec", "panel", "registry"):
        _declared_root_binding(
            output,
            root,
            commit,
            development.get(name),
            label=f"model-suite development {name}",
        )
    _collect_development_bridge(output, root, commit, development)
    _collect_preopening_receipts(output, root, commit, suite)
    suite_source_sha256 = str(development.get("source_sha256", ""))
    if len(suite_source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in suite_source_sha256
    ):
        raise ChronologyError("model suite lacks its source-tree SHA-256")
    versioned = suite.get("versioned_suite")
    if versioned is not None:
        versioned_path = _declared_root_binding(
            output, root, commit, versioned, label="versioned model suite"
        )
        versioned_document = _json_from_git(
            root, commit, versioned_path, label="versioned model suite"
        )
        alias_without_pointer = dict(suite)
        alias_without_pointer.pop("versioned_suite", None)
        if versioned_document != alias_without_pointer:
            raise ChronologyError(
                "versioned model suite differs from the direct opening registry"
            )

    cohorts = suite.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {"temporal", "external"}:
        raise ChronologyError("model suite lacks exact temporal/external cohorts")
    learned_count = 0
    for cohort_name, cohort in cohorts.items():
        entries = cohort.get("models") if isinstance(cohort, Mapping) else None
        if not isinstance(entries, list) or not entries:
            raise ChronologyError(f"{cohort_name} model registry is empty")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ChronologyError("model-suite entry is not an object")
            executor = str(entry.get("executor", ""))
            if executor == "builtin":
                if "artifact" in entry:
                    raise ChronologyError("builtin model unexpectedly binds an artifact")
                continue
            artifact = entry.get("artifact")
            if not isinstance(artifact, Mapping) or "path" not in artifact:
                raise ChronologyError("learned model lacks an artifact path")
            learned_count += 1
            if executor == "lightgbm_bundle":
                manifest_path = _declared_root_binding(
                    output,
                    root,
                    commit,
                    artifact,
                    label="LightGBM manifest",
                )
                _collect_lightgbm_bundle(output, root, commit, manifest_path)
            elif executor in {"thermoroute_bundle", "lstm_bundle"}:
                if not {"metadata_sha256", "weights_sha256"} <= set(artifact):
                    raise ChronologyError("Torch bundle lacks exact metadata/weights hashes")
                directory = _normalise_path(str(artifact["path"]))
                metadata_path = _normalise_path(f"{directory}/metadata.json")
                weights_path = _normalise_path(f"{directory}/weights.pt")
                _add_binding(
                    output,
                    root,
                    commit,
                    metadata_path,
                    expected_sha256=artifact.get("metadata_sha256"),
                )
                _add_binding(
                    output,
                    root,
                    commit,
                    weights_path,
                    expected_sha256=artifact.get("weights_sha256"),
                )
                metadata = _json_from_git(
                    root, commit, metadata_path, label="Torch bundle metadata"
                )
                weights_digest = output[weights_path]["sha256"]
                if metadata.get("weights_sha256") != weights_digest:
                    raise ChronologyError("Torch metadata does not bind weights.pt")
                _collect_prediction_binding(
                    output,
                    root,
                    commit,
                    metadata.get("development_prediction"),
                    label=str(entry.get("model_id", executor)),
                )
            else:
                raise ChronologyError(f"unsafe model-suite executor: {executor}")
    if learned_count < 1:
        raise ChronologyError("model suite contains no learned artifacts")

    _add_binding(output, root, commit, replay_path)
    replay = _json_from_git(root, commit, replay_path, label="development replay receipt")
    if replay.get("format") != REPLAY_FORMAT:
        raise ChronologyError("unsupported development replay receipt")
    replay_suite = replay.get("suite")
    if not isinstance(replay_suite, Mapping):
        raise ChronologyError("development replay does not bind its model suite")
    if (
        _normalise_path(str(replay_suite.get("path", ""))) != suite_path
        or replay_suite.get("sha256") != output[suite_path]["sha256"]
    ):
        raise ChronologyError("development replay binds another model suite")
    replay_self = replay.get("receipt_self_sha256")
    if replay_self is not None:
        unhashed = dict(replay)
        unhashed.pop("receipt_self_sha256", None)
        if replay_self != _repro_sha256_json(unhashed):
            raise ChronologyError("development replay self-hash changed")
    replay_source_sha256 = str(replay.get("source_tree_sha256", ""))
    if len(replay_source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in replay_source_sha256
    ):
        raise ChronologyError("development replay lacks its source-tree SHA-256")
    return (
        [output[path] for path in sorted(output)],
        suite_source_sha256,
        replay_source_sha256,
    )


def _collect_snapshot_files(
    output: dict[str, dict[str, Any]],
    root: Path,
    commit: str,
    index_path: str,
    *,
    require_metadata_binding: bool = False,
) -> set[str]:
    document = _json_from_git(root, commit, index_path, label="raw snapshot index")
    records = document.get("records")
    if (
        not isinstance(records, list) or not records
        or (
            require_metadata_binding
            and (
                set(document) != {"schema_version", "snapshot_count", "records"}
                or document.get("schema_version") != 2
                or type(document.get("snapshot_count")) is not int
                or document["snapshot_count"] != len(records)
            )
        )
    ):
        raise ChronologyError(f"snapshot index is empty or malformed: {index_path}")
    linked: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ChronologyError("snapshot-index record is not an object")
        if require_metadata_binding and (
            set(record) != {
                "provider", "request_sha256", "response_sha256",
                "metadata_sha256", "metadata_byte_count", "retrieved_at_utc",
                "byte_count", "request", "metadata_path", "response_path",
            }
            or not isinstance(record.get("metadata_sha256"), str)
            or len(record["metadata_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record["metadata_sha256"]
            )
            or type(record.get("metadata_byte_count")) is not int
            or record["metadata_byte_count"] < 1
        ):
            raise ChronologyError("snapshot-index metadata binding is malformed")
        for field in ("metadata_path", "response_path"):
            if field not in record:
                raise ChronologyError(f"snapshot record lacks {field}")
            path = _join_relative(index_path, record[field])
            expected = record.get(
                "response_sha256" if field == "response_path" else "metadata_sha256"
            )
            _add_binding(output, root, commit, path, expected_sha256=expected)
            expected_bytes = record.get(
                "byte_count" if field == "response_path" else "metadata_byte_count"
            )
            if require_metadata_binding and output[path]["byte_count"] != expected_bytes:
                raise ChronologyError(f"snapshot {field} byte count changed")
            linked.add(path)
    return linked


def _collect_input_evidence(
    root: Path,
    commit: str,
    *,
    candidate_table: str,
    candidate_provenance: str,
    candidate_snapshot_index: str,
    external_registry: str,
    external_lock: str,
    input_manifest: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    output: dict[str, dict[str, Any]] = {}
    new_paths: set[str] = set()
    for path in (
        candidate_table,
        candidate_provenance,
        candidate_snapshot_index,
        external_registry,
        external_lock,
        input_manifest,
    ):
        _add_binding(output, root, commit, path)
        new_paths.add(path)

    lock = _json_from_git(root, commit, external_lock, label="external registry lock")
    if lock.get("status") != "REGISTRY_FROZEN_LABELS_SEALED":
        raise ChronologyError("external registry lock is not sealed")
    expected_registry_digest = output[external_registry]["sha256"]
    if lock.get("confirmatory_registry_sha256") != expected_registry_digest:
        raise ChronologyError("external registry lock binds another registry")
    frozen = lock.get("frozen_artifacts")
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "development_panel_spec",
        "candidate_table",
        "candidate_provenance",
        "candidate_snapshot_index",
    }:
        raise ChronologyError("external registry lock lacks exact frozen artifacts")
    expected_candidate_paths = {
        "candidate_table": candidate_table,
        "candidate_provenance": candidate_provenance,
        "candidate_snapshot_index": candidate_snapshot_index,
    }
    for name, expected_path in expected_candidate_paths.items():
        actual_path = _declared_root_binding(
            output,
            root,
            commit,
            frozen.get(name),
            label=f"external lock {name}",
        )
        if actual_path != expected_path:
            raise ChronologyError(f"external lock names another {name}")
    # The development-panel specification is supporting selection evidence.  It
    # is expected to predate model freezing, so it is bound but not classified
    # as newly acquired confirmation evidence.
    _declared_root_binding(
        output,
        root,
        commit,
        frozen["development_panel_spec"],
        label="external lock development-panel spec",
    )
    new_paths |= _collect_snapshot_files(
        output, root, commit, candidate_snapshot_index
    )

    manifest = _json_from_git(root, commit, input_manifest, label="actual-input manifest")
    if (
        manifest.get("format") != INPUT_MANIFEST_FORMAT
        or manifest.get("status") != "FROZEN_PRELABEL_NO_OUTCOMES"
        or manifest.get("contains_outcome") is not False
        or manifest.get("contains_outcome_labels") is not False
        or manifest.get("post_2020_wtemp_requested_or_inspected") is not False
    ):
        raise ChronologyError("actual-input manifest is not sealed outcome-free evidence")
    cohort_tables = manifest.get("cohort_tables")
    if not isinstance(cohort_tables, Mapping) or set(cohort_tables) != {
        "temporal",
        "external",
    }:
        raise ChronologyError("actual-input manifest lacks exact normalized cohorts")
    for cohort, value in cohort_tables.items():
        path = _declared_root_binding(
            output,
            root,
            commit,
            value,
            label=f"{cohort} normalized input table",
        )
        new_paths.add(path)

    registries = manifest.get("registry_inputs")
    if not isinstance(registries, Mapping) or set(registries) != {"temporal", "external"}:
        raise ChronologyError("actual-input manifest lacks exact registry inputs")
    for cohort, value in registries.items():
        path = _declared_root_binding(
            output,
            root,
            commit,
            value,
            label=f"{cohort} input registry",
        )
        if cohort == "external" and path != external_registry:
            raise ChronologyError("actual-input manifest uses another external registry")

    source_evidence = manifest.get("source_evidence")
    if not isinstance(source_evidence, list) or not source_evidence:
        raise ChronologyError("actual-input manifest lacks raw/normalized source evidence")
    for index, item in enumerate(source_evidence):
        if not isinstance(item, Mapping):
            raise ChronologyError("actual-input source evidence is not an object")
        if (
            item.get("contains_outcome") is not False
            or item.get("contains_outcome_labels") is not False
        ):
            raise ChronologyError("actual-input source evidence does not exclude outcomes")
        path = _declared_root_binding(
            output,
            root,
            commit,
            item.get("artifact"),
            label=f"actual-input source evidence {index}",
        )
        new_paths.add(path)
        if item.get("evidence_type") == "snapshot_index":
            new_paths |= _collect_snapshot_files(output, root, commit, path)
        elif item.get("evidence_type") != "normalized_immutable_snapshot":
            raise ChronologyError("unsupported actual-input evidence type")
    return [output[path] for path in sorted(output)], new_paths


def _protocol_history(
    root: Path,
    *,
    seal_path: str,
    model_commit: str,
) -> dict[str, Any]:
    seal_path = _normalise_path(seal_path)
    seal_binding = _binding(root, model_commit, seal_path)
    seal = _json_from_git(root, model_commit, seal_path, label="protocol seal")
    if (
        seal.get("format") != PROTOCOL_SEAL_FORMAT
        or seal.get("status") != "SEALED_PRELABEL_OUTCOMES_NOT_ACQUIRED"
    ):
        raise ChronologyError("unsupported Route-A protocol seal")
    original = seal.get("original_preregistration")
    final = seal.get("final_prelabel_protocol")
    if not isinstance(original, Mapping) or not isinstance(final, Mapping):
        raise ChronologyError("protocol seal lacks original/final history")
    original_commit = _resolve_commit(root, str(original.get("commit", "")))
    final_commit = _resolve_commit(root, str(final.get("commit", "")))
    _strictly_precedes(
        root,
        original_commit,
        final_commit,
        label="original protocol < final pre-label protocol",
    )
    _strictly_precedes(
        root,
        final_commit,
        model_commit,
        label="final pre-label protocol < model freezing",
    )
    declared: list[dict[str, Any]] = []
    original_markdown = original.get("markdown")
    if not isinstance(original_markdown, Mapping):
        raise ChronologyError("protocol seal lacks original Markdown binding")
    original_path = _normalise_path(str(original_markdown.get("path", "")))
    original_payload = _git_show(root, original_commit, original_path)
    if _sha256_bytes(original_payload) != original_markdown.get("sha256"):
        raise ChronologyError("original protocol Markdown differs from its seal")
    declared.append(
        {
            "role": "original_markdown",
            "commit": original_commit,
            "path": original_path,
            "sha256": _sha256_bytes(original_payload),
        }
    )
    for role in ("json", "markdown"):
        value = final.get(role)
        if not isinstance(value, Mapping):
            raise ChronologyError(f"protocol seal lacks final {role} binding")
        path = _normalise_path(str(value.get("path", "")))
        payload = _git_show(root, final_commit, path)
        digest = _sha256_bytes(payload)
        if digest != value.get("sha256"):
            raise ChronologyError(f"final protocol {role} differs from its seal")
        current = root / path
        if current.is_symlink() or not current.is_file() or current.read_bytes() != payload:
            raise ChronologyError(f"current final protocol {role} bytes changed")
        declared.append(
            {
                "role": f"final_{role}",
                "commit": final_commit,
                "path": path,
                "sha256": digest,
            }
        )
    return {
        "seal": seal_binding,
        "original_commit": original_commit,
        "final_prelabel_commit": final_commit,
        "declared_git_show_bindings": declared,
    }


def _is_protected(path: str) -> bool:
    path = _normalise_path(path)
    if path in PROTECTED_EXACT_FILES:
        return True
    if any(path == directory or path.startswith(f"{directory}/") for directory in PROTECTED_DIRECTORIES):
        return True
    if "/" not in path and any(
        fnmatch.fnmatchcase(path, pattern) for pattern in PROTECTED_ROOT_PATTERNS
    ):
        return True
    return False


def _git_model_control_paths(root: Path, commit: str) -> set[str]:
    output = _git(root, "ls-tree", "-r", "--name-only", "-z", commit)
    return {
        value.decode("utf-8", errors="strict")
        for value in output.split(b"\0")
        if value and _is_protected(value.decode("utf-8", errors="strict"))
    }


def _working_model_control_paths(root: Path) -> set[str]:
    output: set[str] = set()
    for directory in PROTECTED_DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            output.add(directory)
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                raise ChronologyError(
                    "compiled Python cache is prohibited in protected paths: "
                    f"{relative}"
                )
            if path.is_file() or path.is_symlink():
                output.add(relative)
    for relative in PROTECTED_EXACT_FILES:
        path = root / relative
        if path.exists() or path.is_symlink():
            output.add(relative)
    for path in root.iterdir():
        if path.is_file() or path.is_symlink():
            relative = path.name
            if any(
                fnmatch.fnmatchcase(relative, pattern)
                for pattern in PROTECTED_ROOT_PATTERNS
            ):
                output.add(relative)
    return output


def _collect_model_source_control(
    root: Path,
    model_commit: str,
) -> tuple[list[dict[str, Any]], str]:
    """Bind the exact tracked source/control filesystem at model freezing."""
    tracked = _git_model_control_paths(root, model_commit)
    working = _working_model_control_paths(root)
    if tracked != working:
        raise ChronologyError(
            "working source/control path set differs from model-freeze Git tree: "
            f"missing={sorted(tracked - working)[:10]}, "
            f"untracked_or_ignored={sorted(working - tracked)[:10]}"
        )
    bindings = [_binding(root, model_commit, path) for path in sorted(tracked)]
    binding_by_path = {item["path"]: item for item in bindings}
    source_paths: set[str] = set()
    for pattern in SOURCE_INVENTORY_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                source_paths.add(path.relative_to(root).as_posix())
    if not source_paths or not source_paths <= tracked:
        raise ChronologyError("model source inventory is empty or leaves control closure")
    inventory = {
        path: binding_by_path[path]["sha256"] for path in sorted(source_paths)
    }
    return bindings, _repro_sha256_json(inventory)


def _commit_touches(root: Path, commit: str) -> set[str]:
    output = _git(
        root,
        "diff-tree",
        "--root",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-m",
        "-z",
        commit,
    )
    return {
        item.decode("utf-8")
        for item in output.split(b"\0")
        if item
    }


def _path_creation_commits(root: Path, head: str, path: str) -> list[str]:
    """Return every reachable commit that creates ``path`` from no parent copy.

    This deliberately avoids path-limited ``git log`` history simplification.
    A receipt added and deleted on a merged side branch remains visible, while
    a merge that merely carries an already-existing path from one parent is not
    misclassified as a second creation.
    """
    path = _normalise_path(path)
    commits = [
        value
        for value in _git(root, "rev-list", "--reverse", head).decode().splitlines()
        if value
    ]
    creations: list[str] = []
    for commit in commits:
        if not _git_path_exists(root, commit, path):
            continue
        lineage = _git(root, "rev-list", "--parents", "-n", "1", commit).decode()
        fields = lineage.strip().split()
        if not fields or fields[0] != commit:
            raise ChronologyError("cannot replay receipt path history")
        parents = fields[1:]
        if not any(_git_path_exists(root, parent, path) for parent in parents):
            creations.append(commit)
    return creations


def _touched_after(
    root: Path,
    start_exclusive: str,
    end_inclusive: str,
    *,
    predicate: Any,
) -> list[dict[str, Any]]:
    commits = [
        item
        for item in _git(
            root, "rev-list", "--reverse", f"{start_exclusive}..{end_inclusive}"
        ).decode().splitlines()
        if item
    ]
    output: list[dict[str, Any]] = []
    for commit in commits:
        paths = sorted(path for path in _commit_touches(root, commit) if predicate(path))
        if paths:
            output.append({"commit": commit, "paths": paths})
    return output


def _worktree_protected_changes(root: Path) -> list[str]:
    tracked = {
        item.decode("utf-8")
        for item in _git(root, "diff", "--name-only", "-z", "HEAD", "--").split(b"\0")
        if item
    }
    untracked = {
        item.decode("utf-8")
        for item in _git(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        if item
    }
    return sorted(path for path in tracked | untracked if _is_protected(path))


def _assert_no_artifact_touches(
    root: Path,
    *,
    start: str,
    end: str,
    paths: Iterable[str],
    label: str,
) -> None:
    protected = set(paths)
    touches = _touched_after(
        root, start, end, predicate=lambda candidate: candidate in protected
    )
    if touches:
        raise ChronologyError(f"{label} changed after its freeze commit: {touches[:3]}")


def _assert_absent_at_model_freeze(
    root: Path,
    commit: str,
    paths: Iterable[str],
) -> list[str]:
    checked = sorted({_normalise_path(path) for path in paths})
    present = [path for path in checked if _git_tree_paths(root, commit, path)]
    if present:
        raise ChronologyError(
            "candidate/confirmation-period input artifacts existed at model freeze: "
            f"{present[:10]}"
        )
    return checked


def _require_gate_files_at_model(
    root: Path, model_commit: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in REQUIRED_GATE_PATHS:
        if not _git_path_exists(root, model_commit, path):
            raise ChronologyError(
                f"chronology implementation was not committed before model freeze: {path}"
            )
        output.append(_binding(root, model_commit, path))
    return output


def _normalise_paths(paths: Mapping[str, str | Path]) -> dict[str, str]:
    required = {
        "protocol_seal",
        "model_suite",
        "development_replay",
        "candidate_table",
        "candidate_provenance",
        "candidate_snapshot_index",
        "external_registry",
        "external_lock",
        "input_manifest",
    }
    if set(paths) != required:
        raise ChronologyError(
            f"chronology path registry changed: expected={sorted(required)}, found={sorted(paths)}"
        )
    return {name: _normalise_path(str(value)) for name, value in paths.items()}


def _evaluate_chronology(
    root: Path,
    *,
    model_commit: str,
    evidence_commit: str,
    creation_base_commit: str,
    current_head: str,
    paths: Mapping[str, str],
) -> dict[str, Any]:
    assert_no_hidden_index_flags(root)
    _strictly_precedes(
        root, model_commit, evidence_commit, label="model freeze < input evidence"
    )
    _strictly_precedes(
        root,
        evidence_commit,
        creation_base_commit,
        label="input evidence < chronology receipt creation base",
    )
    if not _is_ancestor(root, creation_base_commit, current_head):
        raise ChronologyError("chronology receipt creation base is not an ancestor of HEAD")

    protocol = _protocol_history(
        root,
        seal_path=paths["protocol_seal"],
        model_commit=model_commit,
    )
    gate_files = _require_gate_files_at_model(root, model_commit)
    source_control, source_tree_sha256 = _collect_model_source_control(
        root, model_commit
    )
    (
        model_artifacts,
        suite_source_sha256,
        replay_source_sha256,
    ) = _collect_model_artifacts(
        root,
        model_commit,
        paths["model_suite"],
        paths["development_replay"],
    )
    if {
        source_tree_sha256,
        suite_source_sha256,
        replay_source_sha256,
    } != {source_tree_sha256}:
        raise ChronologyError(
            "model Git source tree, suite lineage and development replay differ"
        )
    input_artifacts, newly_acquired_paths = _collect_input_evidence(
        root,
        evidence_commit,
        candidate_table=paths["candidate_table"],
        candidate_provenance=paths["candidate_provenance"],
        candidate_snapshot_index=paths["candidate_snapshot_index"],
        external_registry=paths["external_registry"],
        external_lock=paths["external_lock"],
        input_manifest=paths["input_manifest"],
    )
    absence_paths = _assert_absent_at_model_freeze(
        root,
        model_commit,
        [*FIXED_PRELABEL_ABSENCE_PATHS, *newly_acquired_paths],
    )

    control_touches = _touched_after(
        root, model_commit, current_head, predicate=_is_protected
    )
    if control_touches:
        raise ChronologyError(
            "source/control path changed after model freeze: "
            f"{control_touches[:3]}"
        )
    worktree_changes = _worktree_protected_changes(root)
    if worktree_changes:
        raise ChronologyError(
            f"uncommitted source/control changes exist: {worktree_changes[:10]}"
        )
    _assert_no_artifact_touches(
        root,
        start=model_commit,
        end=current_head,
        paths=(item["path"] for item in model_artifacts),
        label="model/replay artifact",
    )
    _assert_no_artifact_touches(
        root,
        start=evidence_commit,
        end=current_head,
        paths=(item["path"] for item in input_artifacts),
        label="input-evidence artifact",
    )

    return {
        "format": CHRONOLOGY_FORMAT,
        "status": CHRONOLOGY_STATUS,
        "order": {
            "model_freeze_commit": model_commit,
            "input_evidence_commit": evidence_commit,
            "receipt_creation_base_commit": creation_base_commit,
            "strict_order_verified": True,
        },
        "protocol_history": protocol,
        "paths": dict(paths),
        "required_gate_files_at_model_freeze": gate_files,
        "model_source_control_artifacts": source_control,
        "source_tree_sha256": source_tree_sha256,
        "model_freeze_artifacts": model_artifacts,
        "input_evidence_artifacts": input_artifacts,
        "absence_at_model_freeze": {
            "checked_paths": absence_paths,
            "present_paths": [],
        },
        "post_model_control_audit": {
            "protected_directories": list(PROTECTED_DIRECTORIES),
            "protected_exact_files": list(PROTECTED_EXACT_FILES),
            "protected_root_patterns": list(PROTECTED_ROOT_PATTERNS),
            "committed_touches": [],
            "worktree_changes": [],
        },
        "post_freeze_artifact_mutation_count": 0,
        "external_timestamp_or_public_preregistration": False,
        "independent_custodian_or_worm_storage": False,
        "evidence_scope": EVIDENCE_SCOPE,
        "fallback_if_validation_fails": (
            "TRANSDUCTIVE_RETROSPECTIVE_EXPLORATION_CONFIRMATION_CLAIMS_PROHIBITED"
        ),
    }


def freeze_prelabel_chronology(
    destination: str | Path,
    *,
    root: str | Path,
    model_freeze_commit: str,
    input_evidence_commit: str,
    protocol_seal: str | Path = DEFAULT_PROTOCOL_SEAL,
    model_suite: str | Path = DEFAULT_MODEL_SUITE,
    development_replay: str | Path = DEFAULT_DEVELOPMENT_REPLAY,
    candidate_table: str | Path = DEFAULT_CANDIDATE_TABLE,
    candidate_provenance: str | Path = DEFAULT_CANDIDATE_PROVENANCE,
    candidate_snapshot_index: str | Path = DEFAULT_CANDIDATE_SNAPSHOT_INDEX,
    external_registry: str | Path = DEFAULT_EXTERNAL_REGISTRY,
    external_lock: str | Path = DEFAULT_EXTERNAL_LOCK,
    input_manifest: str | Path = DEFAULT_INPUT_MANIFEST,
) -> dict[str, Any]:
    """Create one immutable chronology receipt after the evidence commit."""
    root = _require_git_root(root)
    destination_path = Path(destination)
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    destination_path = destination_path.resolve()
    destination_relative = _relative(root, destination_path)
    if destination_relative != DEFAULT_RECEIPT:
        raise ChronologyError(
            f"chronology receipt must use its canonical path: {DEFAULT_RECEIPT}"
        )
    if destination_path.exists():
        raise ChronologyError(f"refusing to replace chronology receipt: {destination_path}")
    model_commit = _resolve_commit(root, model_freeze_commit)
    evidence_commit = _resolve_commit(root, input_evidence_commit)
    head = _resolve_commit(root, "HEAD")
    paths = _normalise_paths(
        {
            "protocol_seal": _relative(root, protocol_seal),
            "model_suite": _relative(root, model_suite),
            "development_replay": _relative(root, development_replay),
            "candidate_table": _relative(root, candidate_table),
            "candidate_provenance": _relative(root, candidate_provenance),
            "candidate_snapshot_index": _relative(root, candidate_snapshot_index),
            "external_registry": _relative(root, external_registry),
            "external_lock": _relative(root, external_lock),
            "input_manifest": _relative(root, input_manifest),
        }
    )
    if _git_path_exists(root, evidence_commit, destination_relative):
        raise ChronologyError("chronology receipt already existed at input-evidence commit")
    if _git_path_exists(root, head, destination_relative):
        raise ChronologyError(
            "chronology receipt already existed at its declared creation base"
        )
    stable = _evaluate_chronology(
        root,
        model_commit=model_commit,
        evidence_commit=evidence_commit,
        creation_base_commit=head,
        current_head=head,
        paths=paths,
    )
    document = {**stable, "receipt_self_sha256": _sha256_json(stable)}
    payload = _canonical_json_bytes(document)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as exc:  # race-safe create-only publication
        raise ChronologyError(
            f"refusing to replace chronology receipt: {destination_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve a partial create as evidence of an interrupted attempt.
        raise
    return document


def validate_prelabel_chronology(
    receipt_path: str | Path,
    *,
    root: str | Path,
    allow_gitless_archive: bool = False,
) -> dict[str, Any]:
    """Replay the receipt, using Git unless validating an extracted archive.

    In the live repository this always re-runs ancestry, absence, no-touch and
    exact ``git show`` assertions.  A deliberately gitless release may request
    a current-byte-only replay; the release verifier remains responsible for
    the mandatory Git-bundle ancestry proof.
    """
    root = Path(root).resolve()
    git_available = (root / ".git").exists()
    if git_available or not allow_gitless_archive:
        root = _require_git_root(root)
    path = Path(receipt_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    receipt_relative = _relative(root, path)
    if receipt_relative != DEFAULT_RECEIPT:
        raise ChronologyError(
            f"chronology receipt must use its canonical path: {DEFAULT_RECEIPT}"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChronologyError("chronology receipt is absent or invalid JSON") from exc
    if not isinstance(document, dict) or document.get("format") != CHRONOLOGY_FORMAT:
        raise ChronologyError("unsupported chronology receipt")
    stable = dict(document)
    self_digest = stable.pop("receipt_self_sha256", None)
    if self_digest != _sha256_json(stable):
        raise ChronologyError("chronology receipt self-hash changed")
    order = document.get("order")
    if not isinstance(order, Mapping):
        raise ChronologyError("chronology receipt lacks commit order")
    paths_raw = document.get("paths")
    if not isinstance(paths_raw, Mapping):
        raise ChronologyError("chronology receipt lacks its artifact path registry")
    paths = _normalise_paths({str(key): str(value) for key, value in paths_raw.items()})
    if not git_available and allow_gitless_archive:
        _validate_portable_receipt_bytes(root, document)
        return document
    model_commit = _resolve_commit(root, str(order.get("model_freeze_commit", "")))
    evidence_commit = _resolve_commit(root, str(order.get("input_evidence_commit", "")))
    creation_base = _resolve_commit(
        root, str(order.get("receipt_creation_base_commit", ""))
    )
    current_head = _resolve_commit(root, "HEAD")
    if _git_path_exists(root, evidence_commit, receipt_relative):
        raise ChronologyError("chronology receipt predates its declared evidence commit")
    if _git_path_exists(root, creation_base, receipt_relative):
        raise ChronologyError("chronology receipt existed at its declared creation base")
    _strictly_precedes(
        root,
        creation_base,
        current_head,
        label="chronology receipt creation base < committed receipt",
    )
    if not _git_path_exists(root, current_head, receipt_relative):
        raise ChronologyError("chronology receipt is not committed at HEAD")
    if _git_show(root, current_head, receipt_relative) != path.read_bytes():
        raise ChronologyError("chronology receipt bytes differ from committed HEAD")
    touches = _touched_after(
        root,
        creation_base,
        current_head,
        predicate=lambda candidate: candidate == receipt_relative,
    )
    additions = _path_creation_commits(root, current_head, receipt_relative)
    if (
        len(touches) != 1
        or len(additions) != 1
        or touches[0]["commit"] != additions[0]
    ):
        raise ChronologyError(
            "chronology receipt must be added exactly once after its creation base"
        )
    expected = _evaluate_chronology(
        root,
        model_commit=model_commit,
        evidence_commit=evidence_commit,
        creation_base_commit=creation_base,
        current_head=current_head,
        paths=paths,
    )
    if stable != expected:
        raise ChronologyError("chronology receipt content differs from exact replay")
    return document


__all__ = [
    "CHRONOLOGY_FORMAT",
    "CHRONOLOGY_STATUS",
    "ChronologyError",
    "DEFAULT_CANDIDATE_PROVENANCE",
    "DEFAULT_CANDIDATE_SNAPSHOT_INDEX",
    "DEFAULT_CANDIDATE_TABLE",
    "DEFAULT_DEVELOPMENT_REPLAY",
    "DEFAULT_EXTERNAL_LOCK",
    "DEFAULT_EXTERNAL_REGISTRY",
    "DEFAULT_INPUT_MANIFEST",
    "DEFAULT_MODEL_SUITE",
    "DEFAULT_PROTOCOL_SEAL",
    "DEFAULT_RECEIPT",
    "assert_no_hidden_index_flags",
    "freeze_prelabel_chronology",
    "validate_prelabel_chronology",
]
