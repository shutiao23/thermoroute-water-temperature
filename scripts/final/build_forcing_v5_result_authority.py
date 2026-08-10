#!/usr/bin/env python3
"""Verify and publish the forcing-v5 *point-estimate* result authority.

This verifier is intentionally downstream of, and distinct from, the sealed
score-execution authority.  Its only admissible result input is the fixed
create-only v5 bundle produced by ``run_forcing_ladder_v5_observed.py``.  It
independently audits the exact twelve shards, reconstructs raw-observed label
lineage and station-first point metrics, and deterministically refits one
predeclared raw cell before it may publish a second create-only bundle.

No interval, p-value, bootstrap, sign-flip, or other inferential result is
created here.  Such quantities require a separately frozen and separately
published inference authority.  The default invocation is a byte-free dry run.
Verification and publication require an externally retained SHA256 of the
runner manifest; publication additionally requires an explicit ``--publish``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5  # noqa: E402


class AuthorityError(RuntimeError):
    """A result, chronology, source, or create-only check failed closed."""


AUTHORITY_FORMAT = "thermoroute.forcing-v5-observed-point-result-authority.v1"
AUTHORITY_STATUS = "POST_OUTCOME_OBSERVED_LINEAGE_CORRECTION_POINT_ESTIMATES_ONLY"
INFERENCE_STATUS = "NOT_PUBLISHED_NOT_AUTHORIZED_BY_THIS_AUTHORITY"
DEFAULT_RESULT_DIR = V5.DEFAULT_OUTPUT_DIR
DEFAULT_AUTHORITY_DIR = V5.FINAL_OUTPUT_ROOT / "forcing_regime_v5_observed_point_authority_v1"
RESERVED_INFERENCE_DIR = (
    V5.FINAL_OUTPUT_ROOT / "forcing_regime_v5_observed_inference_authority_v1"
)
BUILDER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests" / "final" / "test_forcing_v5_result_authority.py"

POINT_STATION_FILENAME = "forcing_v5_point_station_metrics.parquet"
POINT_EFFECT_FILENAME = "forcing_v5_point_paired_effects.parquet"
POINT_SUMMARY_FILENAME = "forcing_v5_point_summary.json"
POINT_MANIFEST_FILENAME = "forcing_v5_point_result_authority_manifest.json"

MIN_REPORTABLE_KEYS = 100
RECOMPUTED_CELL = V5.Cell("F0", "LightGBM", 1)
LEGACY_RESULT_NAMES = frozenset(
    {
        "forcing_shards_v4",
        "forcing_regime_v4",
        "forcing_regime_v4_authority",
        "forcing_effects_v4.parquet",
        "forcing_contrasts_v4.parquet",
        "forcing_summary_v4.json",
        "forcing_shards",
    }
)
INFERENCE_TOKENS = (
    "bootstrap",
    "confidence",
    "credible",
    "interval",
    "p_value",
    "pvalue",
    "sign_flip",
    "significance",
)

EXPECTED_MANIFEST_KEYS = frozenset(
    {
        "format",
        "schema_version",
        "analysis_status",
        "execution_performed",
        "scope",
        "captured_inputs_dependencies_and_authorities",
        "governance",
        "raw_panel_lineage",
        "imputed_feature_lineage",
        "preprocessing",
        "training_identity_by_horizon",
        "runtime",
        "publication",
        "cells",
    }
)
EXPECTED_CELL_KEYS = frozenset(
    {
        "cell",
        "analysis_status",
        "shard",
        "features",
        "future_registry",
        "substitution_accounting",
        "fit",
    }
)
EXPECTED_CAPTURE_ROLES = frozenset(
    {
        *V5.PINNED_INPUT_PATHS,
        *V5.PINNED_GOVERNANCE_PATHS,
        *V5.PINNED_DEPENDENCY_PATHS,
        "defect_authority_manifest",
        "defect_authority_report",
        "runner",
        "score_authority_builder",
        "sealed_score_protocol",
        "sealed_score_protocol_seal",
        "score_clean_design_commit",
        "score_source_registry",
    }
)

POINT_STATION_COLUMNS = (
    "authority_format",
    "authority_status",
    "arm",
    "model",
    "horizon",
    "site_id",
    "n_keys",
    "key_set_sha256",
    "key_and_y_sha256",
    "rmse",
    "mae",
    "bias",
    "rmse_damped",
    "mae_damped",
    "bias_damped",
    "forcing_substitution_count",
    *V5.SUBSTITUTION_COLUMNS,
    "reportable",
)

POINT_EFFECT_COLUMNS = (
    "authority_format",
    "authority_status",
    "model",
    "horizon",
    "site_id",
    "n_common_keys",
    "key_set_sha256",
    "key_and_y_sha256",
    "rmse_F0",
    "rmse_F3_full",
    "delta_rmse_F3_minus_F0",
    "forcing_value_rmse_F0_minus_F3",
    "mae_F0",
    "mae_F3_full",
    "delta_mae_F3_minus_F0",
    "forcing_value_mae_F0_minus_F3",
    "bias_F0",
    "bias_F3_full",
    "delta_bias_F3_minus_F0",
    "reportable",
)


@dataclass(frozen=True)
class BoundFile:
    path: Path
    payload: bytes
    sha256: str
    stat_signature: tuple[int, int, int, int, int, int]

    @property
    def binding(self) -> dict[str, object]:
        return {
            "path": self.path.relative_to(ROOT).as_posix(),
            "sha256": self.sha256,
            "bytes": len(self.payload),
            "device": self.stat_signature[0],
            "inode": self.stat_signature[1],
            "mtime_ns": self.stat_signature[4],
            "ctime_ns": self.stat_signature[5],
        }

    @property
    def content_binding(self) -> dict[str, object]:
        return {
            "path": self.path.relative_to(ROOT).as_posix(),
            "sha256": self.sha256,
            "bytes": len(self.payload),
        }


@dataclass(frozen=True)
class ResultSnapshot:
    manifest_bound: BoundFile
    manifest: Mapping[str, object]
    shards: Mapping[V5.Cell, BoundFile]
    frames: Mapping[V5.Cell, pd.DataFrame]
    execution_inputs: Mapping[str, BoundFile]
    reference: pd.DataFrame


@dataclass(frozen=True)
class RecomputeEvidence:
    cell: V5.Cell
    frame: pd.DataFrame
    raw_lineage: Mapping[str, object]
    imputed_lineage: Mapping[str, object]
    training_identity_by_horizon: Mapping[str, object]
    preprocessing: Mapping[str, object]
    fit: Mapping[str, object]


@dataclass(frozen=True)
class PointAuthorityCandidate:
    station_metrics: pd.DataFrame
    paired_effects: pd.DataFrame
    summary: Mapping[str, object]
    manifest_base: Mapping[str, object]
    snapshot: ResultSnapshot
    recomputation: RecomputeEvidence


def expected_cells() -> tuple[V5.Cell, ...]:
    cells = tuple(
        V5.Cell(arm, model, horizon)
        for arm in V5.ARMS
        for model in V5.MODELS
        for horizon in V5.HORIZONS
    )
    if len(cells) != 12 or cells != V5.fixed_cells():
        raise AuthorityError("internal result-authority scope is not the frozen twelve cells")
    return cells


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuthorityError(f"{label} must be one lowercase full SHA256")
    return value


def _canonical_json_bytes(document: object) -> bytes:
    try:
        return (
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise AuthorityError("document is not canonical-JSON serializable") from exc


def _canonical_json_sha256(document: object) -> str:
    return _sha256(_canonical_json_bytes(document))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _parse_canonical_json(bound: BoundFile, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(
            bound.payload,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} is not strict JSON") from exc
    if type(document) is not dict:
        raise AuthorityError(f"{label} is not a JSON object")
    if bound.payload != _canonical_json_bytes(document):
        raise AuthorityError(f"{label} bytes are not canonical JSON")
    return document


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_beneath(path: Path, root: Path, *, label: str) -> Path:
    if ".." in path.parts:
        raise AuthorityError(f"{label} contains a forbidden '..' component")
    absolute = _absolute(path)
    root_absolute = _absolute(root)
    try:
        absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise AuthorityError(f"{label} escapes its approved root") from exc
    try:
        if absolute.resolve(strict=True) != absolute:
            raise AuthorityError(f"{label} traverses a symbolic link")
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be resolved") from exc
    return absolute


def _read_stable_regular(path: Path, *, root: Path, label: str) -> BoundFile:
    absolute = _require_beneath(path, root, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise AuthorityError(f"{label} cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorityError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if signature_before != signature_after or len(payload) != before.st_size:
        raise AuthorityError(f"{label} changed while being captured")
    final = absolute.lstat()
    if (
        stat.S_ISLNK(final.st_mode)
        or not stat.S_ISREG(final.st_mode)
        or (final.st_dev, final.st_ino) != (before.st_dev, before.st_ino)
    ):
        raise AuthorityError(f"{label} path changed while being captured")
    return BoundFile(absolute, payload, _sha256(payload), signature_before)


def _revalidate_bound(bound: BoundFile, *, label: str) -> None:
    current = _read_stable_regular(bound.path, root=ROOT, label=label)
    if (
        current.payload != bound.payload
        or current.sha256 != bound.sha256
        or current.stat_signature != bound.stat_signature
    ):
        raise AuthorityError(f"{label} changed after initial capture")


def _require_exact_mapping_keys(
    value: object,
    expected: Iterable[str],
    *,
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise AuthorityError(f"{label} is not an exact mapping")
    expected_set = set(expected)
    if set(value) != expected_set:
        raise AuthorityError(
            f"{label} key set differs: missing={sorted(expected_set - set(value))}, "
            f"extra={sorted(set(value) - expected_set)}"
        )
    return value


def _binding_subset(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise AuthorityError(f"{label} is not a binding")
    path = value.get("path")
    if type(path) is not str or not path or path.startswith("/") or ".." in Path(path).parts:
        raise AuthorityError(f"{label} path is not canonical repository-relative")
    sha256 = _require_sha256(value.get("sha256"), label=f"{label} SHA256")
    byte_count = value.get("bytes")
    if type(byte_count) is not int or byte_count < 1:
        raise AuthorityError(f"{label} byte count is not a positive integer")
    return {"path": path, "sha256": sha256, "bytes": byte_count}


def _assert_binding(value: object, bound: BoundFile, *, label: str, exact_stat: bool) -> None:
    observed = _binding_subset(value, label=label)
    if observed != bound.content_binding:
        raise AuthorityError(f"{label} content binding differs from captured bytes")
    if exact_stat:
        if type(value) is not dict or value != bound.binding:
            raise AuthorityError(f"{label} stable-file identity differs from captured bytes")


def _assert_shard_binding(
    value: object,
    bound: BoundFile,
    *,
    label: str,
    expected_path: str,
) -> None:
    observed = _binding_subset(value, label=label)
    if observed["path"] != expected_path:
        raise AuthorityError(f"{label} shard path differs from its cell")
    if observed["sha256"] != bound.sha256 or observed["bytes"] != len(bound.payload):
        raise AuthorityError(f"{label} content binding differs from captured bytes")


def _cell_dict(cell: V5.Cell) -> dict[str, object]:
    return {"arm": cell.arm, "model": cell.model, "horizon": cell.horizon}


def _parse_cell(value: object, *, label: str) -> V5.Cell:
    document = _require_exact_mapping_keys(value, ("arm", "model", "horizon"), label=label)
    if type(document["arm"]) is not str or type(document["model"]) is not str:
        raise AuthorityError(f"{label} contains non-string cell identities")
    if type(document["horizon"]) is not int:
        raise AuthorityError(f"{label} horizon is not an integer")
    cell = V5.Cell(document["arm"], document["model"], document["horizon"])
    if cell not in expected_cells():
        raise AuthorityError(f"{label} is outside the frozen twelve-cell scope")
    return cell


def _validate_result_directory_identity(result_dir: Path) -> Path:
    declared = Path(result_dir)
    if declared != DEFAULT_RESULT_DIR or ".." in declared.parts:
        raise AuthorityError(f"result input must be exactly {DEFAULT_RESULT_DIR}")
    absolute = _require_beneath(declared, ROOT, label="formal v5 result directory")
    if absolute.name in LEGACY_RESULT_NAMES or "withdrawn" in absolute.name.lower():
        raise AuthorityError("legacy or withdrawn result directory is forbidden")
    return absolute


def _inventory_result_bundle(result_dir: Path) -> dict[str, Path]:
    directory = _validate_result_directory_identity(result_dir)
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise AuthorityError("formal v5 result directory cannot be inventoried") from exc
    if {path.name for path in entries} != {V5.MANIFEST_FILENAME, V5.SHARD_DIRNAME}:
        raise AuthorityError("formal v5 bundle has a missing or unexpected top-level entry")
    for path in entries:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise AuthorityError("formal v5 bundle contains a symbolic-link entry")
        expected_directory = path.name == V5.SHARD_DIRNAME
        if expected_directory != stat.S_ISDIR(info.st_mode):
            raise AuthorityError("formal v5 bundle entry type differs from its contract")
    shard_dir = directory / V5.SHARD_DIRNAME
    expected_names = {V5.shard_filename(cell) for cell in expected_cells()}
    observed = sorted(shard_dir.iterdir(), key=lambda path: path.name)
    if {path.name for path in observed} != expected_names or len(observed) != 12:
        raise AuthorityError("formal v5 shard inventory is not the exact twelve cells")
    inventory = {V5.MANIFEST_FILENAME: directory / V5.MANIFEST_FILENAME}
    for path in observed:
        if any(token in path.name.lower() for token in ("withdrawn", "legacy", "v4")):
            raise AuthorityError("legacy or withdrawn shard entered the v5 inventory")
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size < 1:
            raise AuthorityError("formal v5 shard is empty, non-regular, or a symlink")
        inventory[f"{V5.SHARD_DIRNAME}/{path.name}"] = path
    return inventory


def _read_parquet(bound: BoundFile, *, label: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(bound.payload))
    except Exception as exc:
        raise AuthorityError(f"{label} is not readable Parquet") from exc


def _strict_dates(series: pd.Series, *, label: str) -> pd.Series:
    try:
        dates = pd.to_datetime(series, errors="raise")
    except Exception as exc:
        raise AuthorityError(f"{label} contains an invalid date") from exc
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise AuthorityError(f"{label} is not an exact non-null daily date")
    return dates.astype("datetime64[ns]")


def _strict_finite(series: pd.Series, *, label: str) -> np.ndarray:
    if isinstance(series.dtype, np.dtype) and not np.issubdtype(series.dtype, np.number):
        raise AuthorityError(f"{label} is not numeric")
    values = series.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise AuthorityError(f"{label} contains a null or non-finite value")
    return values


def _strict_integers(series: pd.Series, *, label: str) -> np.ndarray:
    values = _strict_finite(series, label=label)
    if not np.equal(values, np.floor(values)).all():
        raise AuthorityError(f"{label} contains a non-integer value")
    return values.astype(np.int64)


def _strict_strings(series: pd.Series, *, label: str) -> tuple[str, ...]:
    values = tuple(series.to_numpy(copy=True))
    if any(type(value) is not str or not value or not value.isascii() for value in values):
        raise AuthorityError(f"{label} contains a non-canonical string")
    return values


def validate_shard_independently(
    frame: pd.DataFrame,
    *,
    cell: V5.Cell,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Validate one shard without trusting any runner-produced summary."""

    if tuple(frame.columns) != V5.SHARD_COLUMNS or frame.empty:
        raise AuthorityError("v5 shard schema/order differs or the shard is empty")
    checked = frame.copy(deep=True).reset_index(drop=True)
    keys = _strict_strings(checked["key_id"], label="shard key_id")
    sites = _strict_strings(checked["site_id"], label="shard site_id")
    if any(not re.fullmatch(r"[0-9]{8,15}", site) for site in sites):
        raise AuthorityError("v5 shard contains a non-canonical station identifier")
    issue = _strict_dates(checked["issue_date"], label="shard issue_date")
    target = _strict_dates(checked["target_date"], label="shard target_date")
    horizon = _strict_integers(checked["horizon"], label="shard horizon")
    if not np.all(horizon == cell.horizon) or not target.equals(
        (issue + pd.to_timedelta(cell.horizon, unit="D")).rename("target_date")
    ):
        raise AuthorityError("v5 shard chronology differs from its cell")
    for column, expected in (
        ("arm", cell.arm),
        ("model", cell.model),
        ("analysis_status", V5.STATUS),
    ):
        values = _strict_strings(checked[column], label=f"shard {column}")
        if any(value != expected for value in values):
            raise AuthorityError("v5 shard cell identity differs")
    for column in ("y_true", "y_pred", "y_damped"):
        _strict_finite(checked[column], label=f"shard {column}")
    counts = np.column_stack(
        [
            _strict_integers(checked[column], label=f"shard {column}")
            for column in V5.SUBSTITUTION_COLUMNS
        ]
    )
    total = _strict_integers(
        checked["forcing_substitution_count"], label="shard substitution total"
    )
    if (
        (counts < 0).any()
        or (counts > cell.horizon).any()
        or (total < 0).any()
        or (total > len(V5.METEOROLOGY_VARIABLES) * cell.horizon).any()
        or not np.array_equal(counts.sum(axis=1), total)
    ):
        raise AuthorityError("v5 shard substitution accounting differs")
    if cell.arm == "F0" and np.any(total != 0):
        raise AuthorityError("F0 shard contains forbidden forcing substitutions")
    if len(set(keys)) != len(keys) or checked.duplicated(
        ["site_id", "issue_date", "target_date"]
    ).any():
        raise AuthorityError("v5 shard duplicates a formal identity")

    ref = reference.loc[reference["lead_days"].eq(cell.horizon)].copy()
    expected_columns = ["key_id", "site_id", "issue_date", "target_date", "y_true"]
    joined = ref[expected_columns].merge(
        checked[expected_columns],
        on="key_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_reference", "_shard"),
    )
    if len(joined) != len(ref) or not joined["_merge"].eq("both").all():
        raise AuthorityError("v5 shard key set is not exactly the formal registry")
    for column in ("site_id", "issue_date", "target_date"):
        if not joined[f"{column}_reference"].equals(joined[f"{column}_shard"]):
            raise AuthorityError(f"v5 shard {column} differs from the formal registry")
    reference_y = _strict_finite(joined["y_true_reference"], label="reference y_true")
    shard_y = _strict_finite(joined["y_true_shard"], label="shard y_true")
    if not np.array_equal(reference_y, shard_y):
        raise AuthorityError("v5 shard y_true bytes differ from the formal registry")
    return checked.sort_values(["site_id", "issue_date"], kind="mergesort").reset_index(
        drop=True
    )


def _expected_execution_binding_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        **dict(V5.PINNED_INPUT_PATHS),
        **dict(V5.PINNED_GOVERNANCE_PATHS),
        **dict(V5.PINNED_DEPENDENCY_PATHS),
        "defect_authority_manifest": V5.DEFECT_AUTHORITY_MANIFEST,
        "defect_authority_report": V5.DEFECT_AUTHORITY_REPORT,
        "runner": Path(V5.__file__),
        "score_authority_builder": V5.SCORE_AUTHORITY_BUILDER,
        "sealed_score_protocol": V5.SEALED_SCORE_PROTOCOL,
        "sealed_score_protocol_seal": V5.SEALED_SCORE_PROTOCOL_SEAL,
        "score_clean_design_commit": V5.SCORE_CLEAN_DESIGN_COMMIT,
        "score_source_registry": V5.SCORE_SOURCE_REGISTRY,
    }
    if set(paths) != EXPECTED_CAPTURE_ROLES:
        raise AuthorityError("internal execution-input role inventory differs")
    return paths


def _capture_execution_inputs_from_manifest(
    declared: object,
) -> dict[str, BoundFile]:
    bindings = _require_exact_mapping_keys(
        declared,
        EXPECTED_CAPTURE_ROLES,
        label="runner captured execution inputs",
    )
    expected_paths = _expected_execution_binding_paths()
    captured: dict[str, BoundFile] = {}
    for role in sorted(EXPECTED_CAPTURE_ROLES):
        binding = _binding_subset(bindings[role], label=f"runner input {role}")
        expected_relative = expected_paths[role].relative_to(ROOT).as_posix()
        if binding["path"] != expected_relative:
            raise AuthorityError(f"runner input {role} path differs from its frozen role")
        path = ROOT / str(binding["path"])
        if any(
            forbidden in path.parts
            for forbidden in LEGACY_RESULT_NAMES
        ) or "withdrawn" in path.as_posix().lower():
            raise AuthorityError("legacy or withdrawn output entered the execution capture")
        bound = _read_stable_regular(path, root=ROOT, label=f"runner input {role}")
        _assert_binding(bindings[role], bound, label=f"runner input {role}", exact_stat=True)
        captured[role] = bound
    return captured


def _flatten_source_categories(source: Mapping[str, object]) -> dict[str, dict[str, object]]:
    categories = source.get("authority_categories")
    if type(categories) is not dict:
        raise AuthorityError("score source registry lacks authority categories")
    output: dict[str, dict[str, object]] = {}
    for category, value in categories.items():
        if category == "runtime_authorities":
            if type(value) is not dict or type(value.get("files")) is not dict:
                raise AuthorityError("score source registry runtime category differs")
            role_map = value["files"]
        else:
            if type(value) is not dict:
                raise AuthorityError("score source registry category is not a role mapping")
            role_map = value
        for role, binding in role_map.items():
            if role in output:
                raise AuthorityError("score source registry duplicates an authority role")
            output[role] = _binding_subset(binding, label=f"score source role {role}")
    return output


def _git_output(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(ROOT), *arguments],
            check=True,
            capture_output=True,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityError("Git source/design binding cannot be resolved") from exc
    return completed.stdout


def _validate_historical_design_git(
    clean: Mapping[str, object],
    source: Mapping[str, object],
) -> dict[str, object]:
    git_design = clean.get("git_design_authority")
    if type(git_design) is not dict:
        raise AuthorityError("clean-design record lacks Git design authority")
    identity = source.get("git_design_identity")
    if type(identity) is not dict:
        raise AuthorityError("score source registry lacks Git identity")
    for key in ("object_format", "commit", "tree"):
        if identity.get(key) != git_design.get(key):
            raise AuthorityError("clean-design and source-registry Git identities differ")
    object_format = git_design.get("object_format")
    commit = git_design.get("commit")
    tree = git_design.get("tree")
    if object_format not in {"sha1", "sha256"} or type(commit) is not str or type(tree) is not str:
        raise AuthorityError("historical Git design identity is not canonical")
    observed_tree = _git_output("rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    if observed_tree != tree:
        raise AuthorityError("historical clean-design commit no longer resolves to its tree")
    design_paths = git_design.get("design_paths")
    if type(design_paths) is not dict or not design_paths:
        raise AuthorityError("historical clean-design path registry is empty")
    for role, record in sorted(design_paths.items()):
        if type(record) is not dict:
            raise AuthorityError("historical clean-design path binding differs")
        binding = _binding_subset(record, label=f"historical design {role}")
        oid = record.get("git_blob_oid")
        if type(oid) is not str:
            raise AuthorityError("historical clean-design blob OID is missing")
        observed_oid = _git_output("rev-parse", f"{commit}:{binding['path']}").decode().strip()
        if observed_oid != oid:
            raise AuthorityError("historical design path does not resolve to its sealed blob")
        payload = _git_output("cat-file", "blob", oid)
        if _sha256(payload) != binding["sha256"] or len(payload) != binding["bytes"]:
            raise AuthorityError("historical design blob bytes differ from their SHA256 binding")
    return {
        "object_format": object_format,
        "commit": commit,
        "tree": tree,
        "design_path_count": len(design_paths),
        "historical_commit_checked_without_requiring_current_HEAD": True,
    }


def validate_execution_authority_chain(
    captured: Mapping[str, BoundFile],
) -> dict[str, object]:
    """Validate the historical terminal authority without requiring HEAD rewind."""

    for role in (
        "sealed_score_protocol",
        "sealed_score_protocol_seal",
        "score_clean_design_commit",
        "score_source_registry",
    ):
        if role not in captured:
            raise AuthorityError(f"execution authority chain lacks {role}")
    protocol = _parse_canonical_json(captured["sealed_score_protocol"], label="score protocol")
    seal = _parse_canonical_json(
        captured["sealed_score_protocol_seal"], label="terminal score seal"
    )
    clean = _parse_canonical_json(
        captured["score_clean_design_commit"], label="clean-design record"
    )
    source = _parse_canonical_json(captured["score_source_registry"], label="score source registry")
    if (
        protocol.get("execution_authorized") is not False
        or protocol.get("terminal_seal_required") is not True
        or seal.get("format") != "thermoroute.forcing-v5-observed-execution-seal.v2"
        or seal.get("status") != "TERMINAL_SCORE_EXECUTION_AUTHORITY"
        or seal.get("execution_authorized") is not True
        or source.get("execution_authorized") is not False
        or clean.get("status") != "CLEAN_DESIGN_COMMITTED_BEFORE_SCORE_EXECUTION"
    ):
        raise AuthorityError("forcing-v5 execution authority chronology/semantics differ")
    expected_scope = V5._score_execution_scope_record()
    for document, label in ((protocol, "protocol"), (seal, "seal"), (clean, "clean design")):
        if document.get("forcing_v5_observed_execution_scope") != expected_scope:
            raise AuthorityError(f"{label} execution scope is not the exact twelve cells")
    for field, role in (
        ("protocol", "sealed_score_protocol"),
        ("runner", "runner"),
        ("clean_design_commit", "score_clean_design_commit"),
        ("source_registry", "score_source_registry"),
    ):
        _assert_binding(
            seal.get(field),
            captured[role],
            label=f"terminal seal {field}",
            exact_stat=False,
        )
    _assert_binding(
        clean.get("runner"), captured["runner"], label="clean-design runner", exact_stat=False
    )
    _assert_binding(
        source.get("runner"), captured["runner"], label="source-registry runner", exact_stat=False
    )
    _assert_binding(
        source.get("clean_design_commit"),
        captured["score_clean_design_commit"],
        label="source-registry clean design",
        exact_stat=False,
    )
    flattened = _flatten_source_categories(source)
    expected_category_roles = EXPECTED_CAPTURE_ROLES - {
        "runner",
        "sealed_score_protocol",
        "sealed_score_protocol_seal",
        "score_clean_design_commit",
        "score_source_registry",
    }
    if set(flattened) != expected_category_roles:
        raise AuthorityError("score source registry does not bind every pre-score role exactly")
    for role, binding in flattened.items():
        _assert_binding(binding, captured[role], label=f"source category {role}", exact_stat=False)
    categories = source["authority_categories"]
    if source.get("authority_categories_sha256") != _canonical_json_sha256(categories):
        raise AuthorityError("score source registry category hash differs")
    roots = seal.get("authority_roots")
    if type(roots) is not dict or set(roots) != set(categories):
        raise AuthorityError("terminal seal authority roots differ")
    expected_roots = {
        category: _canonical_json_sha256(value)
        for category, value in sorted(categories.items())
    }
    if roots != expected_roots or seal.get("authority_roots_sha256") != _canonical_json_sha256(
        expected_roots
    ):
        raise AuthorityError("terminal seal authority-root hashes differ")
    git = _validate_historical_design_git(clean, source)
    return {
        "terminal_execution_authorized": True,
        "protocol_sha256": captured["sealed_score_protocol"].sha256,
        "terminal_seal_sha256": captured["sealed_score_protocol_seal"].sha256,
        "clean_design_sha256": captured["score_clean_design_commit"].sha256,
        "source_registry_sha256": captured["score_source_registry"].sha256,
        "historical_git_design": git,
    }


def _validate_manifest_semantics(manifest: Mapping[str, object]) -> dict[V5.Cell, object]:
    _require_exact_mapping_keys(manifest, EXPECTED_MANIFEST_KEYS, label="runner manifest")
    if (
        manifest.get("format") != V5.MANIFEST_FORMAT
        or manifest.get("schema_version") != V5.SCHEMA_VERSION
        or manifest.get("analysis_status") != V5.STATUS
        or manifest.get("execution_performed") is not True
    ):
        raise AuthorityError("runner manifest identity/status differs")
    scope = _require_exact_mapping_keys(
        manifest.get("scope"),
        ("arms", "models", "horizons", "cell_count", "other_arms_authorized"),
        label="runner manifest scope",
    )
    if scope != {
        "arms": list(V5.ARMS),
        "models": list(V5.MODELS),
        "horizons": list(V5.HORIZONS),
        "cell_count": 12,
        "other_arms_authorized": False,
    }:
        raise AuthorityError("runner manifest scope is not the exact frozen twelve cells")
    publication = manifest.get("publication")
    if type(publication) is not dict or publication != {
        "destination": DEFAULT_RESULT_DIR.relative_to(ROOT).as_posix(),
        "create_only": True,
        "resume": False,
        "overwrite": False,
        "same_parent_staging": True,
        "exclusive_lock": True,
        "all_files_fsynced_before_commit": True,
        "single_linux_RENAME_NOREPLACE": True,
    }:
        raise AuthorityError("runner result publication is not the fixed create-only transaction")
    cells_raw = manifest.get("cells")
    if type(cells_raw) is not list or len(cells_raw) != 12:
        raise AuthorityError("runner manifest does not register exactly twelve cells")
    output: dict[V5.Cell, object] = {}
    for index, value in enumerate(cells_raw):
        row = _require_exact_mapping_keys(
            value, EXPECTED_CELL_KEYS, label=f"runner manifest cell {index}"
        )
        cell = _parse_cell(row["cell"], label=f"runner manifest cell identity {index}")
        if cell in output:
            raise AuthorityError("runner manifest duplicates a cell")
        if row["analysis_status"] != V5.STATUS:
            raise AuthorityError("runner manifest cell analysis status differs")
        output[cell] = row
    if set(output) != set(expected_cells()):
        raise AuthorityError("runner manifest cell inventory differs from the exact twelve cells")
    return output


def _validate_manifest_governance(
    manifest: Mapping[str, object],
    captured: Mapping[str, BoundFile],
) -> None:
    governance = manifest.get("governance")
    if type(governance) is not dict:
        raise AuthorityError("runner manifest lacks governance bindings")
    exact_hashes = {
        "draft_protocol_sha256": V5.PINNED_GOVERNANCE_SHA256["protocol"],
        "key_authority_manifest_sha256": V5.PINNED_GOVERNANCE_SHA256[
            "key_authority_manifest"
        ],
        "defect_authority_manifest_sha256": V5.EXPECTED_DEFECT_AUTHORITY_SHA256,
        "sealed_score_protocol_sha256": captured["sealed_score_protocol"].sha256,
        "sealed_score_protocol_seal_sha256": captured["sealed_score_protocol_seal"].sha256,
        "clean_design_commit_sha256": captured["score_clean_design_commit"].sha256,
        "source_registry_sha256": captured["score_source_registry"].sha256,
    }
    for field, expected in exact_hashes.items():
        if governance.get(field) != expected:
            raise AuthorityError(f"runner manifest governance hash differs: {field}")
    required_flags = {
        "sealed_score_execution_authorized": True,
        "defect_authority_is_score_authorization": False,
        "no_self_hash_cycle": True,
        "precommit_full_snapshot_revalidation_required": True,
    }
    for field, expected in required_flags.items():
        if governance.get(field) is not expected:
            raise AuthorityError(f"runner manifest governance flag differs: {field}")
    if manifest.get("runtime") != V5._runtime_evidence():
        raise AuthorityError("current verifier runtime differs from the scoring runtime")


def _validate_cell_manifest_contract(
    row: Mapping[str, object],
    *,
    cell: V5.Cell,
    frame: pd.DataFrame,
) -> None:
    features = row.get("features")
    if type(features) is not dict:
        raise AuthorityError("runner manifest cell lacks a feature contract")
    expected_columns = (
        V5.FROZEN_BASE_FEATURE_COLUMNS
        if cell.arm == "F0"
        else (*V5.FROZEN_BASE_FEATURE_COLUMNS, *V5._future_feature_columns(cell.horizon))
    )
    if features != {
        "count": len(expected_columns),
        "ordered_columns": list(expected_columns),
        "ordered_columns_sha256": V5._canonical_json_sha256(list(expected_columns)),
    }:
        raise AuthorityError("runner manifest cell feature contract differs")
    future = row.get("future_registry")
    if cell.arm == "F0":
        if future is not None:
            raise AuthorityError("F0 manifest cell binds a forbidden future registry")
    elif (
        type(future) is not dict
        or set(future) != {"source_sha256", "content_sha256", "raw_value_mask_pairs_preserved"}
        or future.get("raw_value_mask_pairs_preserved") is not True
    ):
        raise AuthorityError("F3 manifest cell lacks an exact raw-future lineage")
    else:
        _require_sha256(future.get("source_sha256"), label="future registry source")
        _require_sha256(future.get("content_sha256"), label="future registry content")

    substitutions = row.get("substitution_accounting")
    if type(substitutions) is not dict:
        raise AuthorityError("runner manifest cell lacks substitution accounting")
    totals = {
        column: int(frame[column].sum()) for column in V5.SUBSTITUTION_COLUMNS
    }
    total = int(frame["forcing_substitution_count"].sum())
    if substitutions != {
        "per_variable_totals": totals,
        "total": total,
        "total_equals_sum_per_variable": True,
        "per_row_upper_bound": len(V5.METEOROLOGY_VARIABLES) * cell.horizon,
        "F0_identically_zero": cell.arm == "F0",
    }:
        raise AuthorityError("runner manifest cell substitution totals differ from the shard")

    fit = row.get("fit")
    if type(fit) is not dict:
        raise AuthorityError("runner manifest cell lacks fit evidence")
    expected_params = V5._resolved_lightgbm_params(cell.horizon)
    expected_scalars = {
        "training_rows": V5.EXPECTED_EXAMPLE_COUNTS["train"][cell.horizon],
        "validation_rows": V5.EXPECTED_EXAMPLE_COUNTS["val"][cell.horizon],
        "evaluation_rows": V5.EXPECTED_REFERENCE_COUNTS[cell.horizon],
        "validation_role": "actual_lgb_eval_set_not_training_tail",
        "early_stopping_rounds": 50,
        "best_iteration_upper_bound": V5.BEST_ITER_UPPER_BOUND[cell.horizon],
        "prediction_num_threads": 1,
        "resolved_lightgbm_parameters": expected_params,
    }
    for field, expected in expected_scalars.items():
        if fit.get(field) != expected:
            raise AuthorityError(f"runner manifest cell fit contract differs: {field}")
    best_iteration = fit.get("best_iteration")
    if (
        type(best_iteration) is not int
        or best_iteration < 1
        or best_iteration > V5.BEST_ITER_UPPER_BOUND[cell.horizon]
    ):
        raise AuthorityError("runner manifest cell best iteration lies outside its frozen bound")
    datasets = fit.get("datasets")
    if type(datasets) is not dict or set(datasets) != {"train", "validation", "evaluation"}:
        raise AuthorityError("runner manifest cell dataset evidence differs")
    for name, expected_rows in (
        ("train", V5.EXPECTED_EXAMPLE_COUNTS["train"][cell.horizon]),
        ("validation", V5.EXPECTED_EXAMPLE_COUNTS["val"][cell.horizon]),
        ("evaluation", V5.EXPECTED_REFERENCE_COUNTS[cell.horizon]),
    ):
        dataset = datasets[name]
        required = {
            "rows",
            "ordered_identity_sha256",
            "ordered_identity_and_y_sha256",
            "ordered_y_sha256",
            "actual_feature_matrix_sha256",
            "ordered_feature_columns_sha256",
        }
        if type(dataset) is not dict or set(dataset) != required or dataset.get("rows") != expected_rows:
            raise AuthorityError(f"runner manifest cell {name} dataset binding differs")
        for field in required - {"rows"}:
            _require_sha256(dataset.get(field), label=f"cell {name} {field}")
    outcomes = fit.get("fitted_outcomes")
    if (
        type(outcomes) is not dict
        or set(outcomes)
        != {"train_fitted_outcome_sha256", "validation_fitted_outcome_sha256", "target_kind"}
        or outcomes.get("target_kind")
        != ("damped_residual" if cell.model == "ResidualLightGBM" else "raw_y")
    ):
        raise AuthorityError("runner manifest cell fitted-outcome contract differs")
    _require_sha256(outcomes.get("train_fitted_outcome_sha256"), label="train fitted outcome")
    _require_sha256(
        outcomes.get("validation_fitted_outcome_sha256"), label="validation fitted outcome"
    )


def capture_result_snapshot(
    *,
    expected_manifest_sha256: str,
    result_dir: Path = DEFAULT_RESULT_DIR,
) -> ResultSnapshot:
    expected_sha = _require_sha256(
        expected_manifest_sha256, label="externally retained result manifest SHA256"
    )
    inventory = _inventory_result_bundle(result_dir)
    manifest_bound = _read_stable_regular(
        inventory[V5.MANIFEST_FILENAME], root=ROOT, label="formal v5 runner manifest"
    )
    if manifest_bound.sha256 != expected_sha:
        raise AuthorityError("formal v5 runner manifest differs from the externally retained pin")
    manifest = _parse_canonical_json(manifest_bound, label="formal v5 runner manifest")
    cell_rows = _validate_manifest_semantics(manifest)
    captured = _capture_execution_inputs_from_manifest(
        manifest["captured_inputs_dependencies_and_authorities"]
    )
    _validate_manifest_governance(manifest, captured)
    validate_execution_authority_chain(captured)
    reference = V5.validate_reference_registry(
        _read_parquet(captured["primary_key_registry"], label="primary key registry"),
        require_production_counts=True,
    )
    frames: dict[V5.Cell, pd.DataFrame] = {}
    shards: dict[V5.Cell, BoundFile] = {}
    for cell in expected_cells():
        row = cell_rows[cell]
        if type(row) is not dict:
            raise AuthorityError("runner manifest cell row differs")
        shard_record = row.get("shard")
        if type(shard_record) is not dict:
            raise AuthorityError("runner manifest cell lacks a shard binding")
        expected_relative = f"{V5.SHARD_DIRNAME}/{V5.shard_filename(cell)}"
        if shard_record.get("path") != expected_relative:
            raise AuthorityError("runner manifest shard path differs from its cell")
        bound = _read_stable_regular(
            inventory[expected_relative], root=ROOT, label=f"formal shard {cell}"
        )
        _assert_shard_binding(
            shard_record,
            bound,
            label=f"runner shard {cell}",
            expected_path=expected_relative,
        )
        frame = validate_shard_independently(
            _read_parquet(bound, label=f"formal shard {cell}"),
            cell=cell,
            reference=reference,
        )
        if shard_record.get("rows") != len(frame):
            raise AuthorityError("runner manifest shard row count differs")
        _validate_cell_manifest_contract(row, cell=cell, frame=frame)
        frames[cell] = frame
        shards[cell] = bound
    _assert_cross_cell_identities(frames)
    return ResultSnapshot(manifest_bound, manifest, shards, frames, captured, reference)


def _key_set_sha256(values: Iterable[str]) -> str:
    ordered = sorted(values)
    return _sha256(("\n".join(ordered) + "\n").encode("ascii"))


def _key_and_y_sha256(frame: pd.DataFrame) -> str:
    rows = frame[["key_id", "y_true"]].sort_values("key_id", kind="mergesort")
    digest = hashlib.sha256()
    digest.update(b"thermoroute-forcing-v5-key-y-v1\0")
    for row in rows.itertuples(index=False):
        encoded = row.key_id.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(np.float64(row.y_true).astype("<f8").tobytes())
    return digest.hexdigest()


def _assert_cross_cell_identities(frames: Mapping[V5.Cell, pd.DataFrame]) -> None:
    if set(frames) != set(expected_cells()):
        raise AuthorityError("cross-cell audit lacks the exact twelve cells")
    for horizon in V5.HORIZONS:
        cells = [cell for cell in expected_cells() if cell.horizon == horizon]
        reference = frames[cells[0]][
            ["key_id", "site_id", "issue_date", "target_date", "y_true", "y_damped"]
        ].sort_values("key_id", kind="mergesort").reset_index(drop=True)
        for cell in cells[1:]:
            candidate = frames[cell][reference.columns].sort_values(
                "key_id", kind="mergesort"
            ).reset_index(drop=True)
            for column in ("key_id", "site_id", "issue_date", "target_date"):
                if not candidate[column].equals(reference[column]):
                    raise AuthorityError("cross-cell formal key identity differs")
            for column in ("y_true", "y_damped"):
                if not np.array_equal(
                    _strict_finite(candidate[column], label=f"cross-cell {column}"),
                    _strict_finite(reference[column], label=f"cross-cell {column}"),
                ):
                    raise AuthorityError(f"cross-cell {column} differs on identical keys")


def station_metrics_from_shards(
    frames: Mapping[V5.Cell, pd.DataFrame],
    *,
    minimum_keys: int = MIN_REPORTABLE_KEYS,
) -> pd.DataFrame:
    if type(minimum_keys) is not int or minimum_keys < 1:
        raise AuthorityError("minimum station key count must be a positive integer")
    _assert_cross_cell_identities(frames)
    rows: list[dict[str, object]] = []
    for cell in expected_cells():
        frame = frames[cell]
        for site, group in frame.groupby("site_id", sort=True):
            ordered = group.sort_values("key_id", kind="mergesort")
            y = _strict_finite(ordered["y_true"], label="station y_true")
            pred = _strict_finite(ordered["y_pred"], label="station y_pred")
            damped = _strict_finite(ordered["y_damped"], label="station y_damped")
            error = pred - y
            damped_error = damped - y
            row: dict[str, object] = {
                "authority_format": AUTHORITY_FORMAT,
                "authority_status": AUTHORITY_STATUS,
                "arm": cell.arm,
                "model": cell.model,
                "horizon": cell.horizon,
                "site_id": site,
                "n_keys": len(ordered),
                "key_set_sha256": _key_set_sha256(ordered["key_id"]),
                "key_and_y_sha256": _key_and_y_sha256(ordered),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "rmse_damped": float(np.sqrt(np.mean(np.square(damped_error)))),
                "mae_damped": float(np.mean(np.abs(damped_error))),
                "bias_damped": float(np.mean(damped_error)),
                "forcing_substitution_count": int(ordered["forcing_substitution_count"].sum()),
                "reportable": len(ordered) >= minimum_keys,
            }
            for column in V5.SUBSTITUTION_COLUMNS:
                row[column] = int(ordered[column].sum())
            rows.append(row)
    output = pd.DataFrame(rows, columns=POINT_STATION_COLUMNS)
    if output.empty or output.duplicated(["arm", "model", "horizon", "site_id"]).any():
        raise AuthorityError("station point-metric matrix is empty or duplicated")
    assert_point_only_frame(output, label="station point metrics")
    return output.sort_values(
        ["arm", "model", "horizon", "site_id"], kind="mergesort"
    ).reset_index(drop=True)


def paired_effects_from_station_metrics(station: pd.DataFrame) -> pd.DataFrame:
    if tuple(station.columns) != POINT_STATION_COLUMNS:
        raise AuthorityError("station point-metric schema differs")
    rows: list[dict[str, object]] = []
    for model in V5.MODELS:
        for horizon in V5.HORIZONS:
            subset = station.loc[
                station["model"].eq(model) & station["horizon"].eq(horizon)
            ].copy()
            f0 = subset.loc[subset["arm"].eq("F0")]
            f3 = subset.loc[subset["arm"].eq("F3_full")]
            paired = f0.merge(
                f3,
                on="site_id",
                how="outer",
                validate="one_to_one",
                suffixes=("_F0", "_F3"),
                indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise AuthorityError("F3-F0 station pairing is not two-sided")
            for row in paired.itertuples(index=False):
                if (
                    row.n_keys_F0 != row.n_keys_F3
                    or row.key_set_sha256_F0 != row.key_set_sha256_F3
                    or row.key_and_y_sha256_F0 != row.key_and_y_sha256_F3
                ):
                    raise AuthorityError("F3-F0 effect is not paired on exact station keys/y")
                delta_rmse = float(row.rmse_F3 - row.rmse_F0)
                delta_mae = float(row.mae_F3 - row.mae_F0)
                rows.append(
                    {
                        "authority_format": AUTHORITY_FORMAT,
                        "authority_status": AUTHORITY_STATUS,
                        "model": model,
                        "horizon": horizon,
                        "site_id": row.site_id,
                        "n_common_keys": int(row.n_keys_F0),
                        "key_set_sha256": row.key_set_sha256_F0,
                        "key_and_y_sha256": row.key_and_y_sha256_F0,
                        "rmse_F0": float(row.rmse_F0),
                        "rmse_F3_full": float(row.rmse_F3),
                        "delta_rmse_F3_minus_F0": delta_rmse,
                        "forcing_value_rmse_F0_minus_F3": -delta_rmse,
                        "mae_F0": float(row.mae_F0),
                        "mae_F3_full": float(row.mae_F3),
                        "delta_mae_F3_minus_F0": delta_mae,
                        "forcing_value_mae_F0_minus_F3": -delta_mae,
                        "bias_F0": float(row.bias_F0),
                        "bias_F3_full": float(row.bias_F3),
                        "delta_bias_F3_minus_F0": float(row.bias_F3 - row.bias_F0),
                        "reportable": bool(row.reportable_F0 and row.reportable_F3),
                    }
                )
    output = pd.DataFrame(rows, columns=POINT_EFFECT_COLUMNS)
    if output.empty or output.duplicated(["model", "horizon", "site_id"]).any():
        raise AuthorityError("paired point-effect matrix is empty or duplicated")
    assert_point_only_frame(output, label="paired point effects")
    return output.sort_values(["model", "horizon", "site_id"], kind="mergesort").reset_index(
        drop=True
    )


def assert_point_only_frame(frame: pd.DataFrame, *, label: str) -> None:
    lowered = [column.lower() for column in frame.columns]
    offenders = [
        column
        for column, lower in zip(frame.columns, lowered, strict=True)
        if any(token in lower for token in INFERENCE_TOKENS)
    ]
    if offenders:
        raise AuthorityError(f"{label} crosses the inference-authority boundary: {offenders}")


def build_point_summary(effects: pd.DataFrame) -> dict[str, object]:
    if tuple(effects.columns) != POINT_EFFECT_COLUMNS:
        raise AuthorityError("paired point-effect schema differs")
    rows: list[dict[str, object]] = []
    for (model, horizon), group in effects.groupby(["model", "horizon"], sort=True):
        reportable = group.loc[group["reportable"]]
        if reportable.empty:
            raise AuthorityError(f"no reportable station for {model}/h{horizon}")
        forcing = reportable["forcing_value_rmse_F0_minus_F3"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "n_reportable_stations": len(reportable),
                "station_set_sha256": _key_set_sha256(reportable["site_id"]),
                "median_station_forcing_value_rmse_F0_minus_F3": float(np.median(forcing)),
                "median_station_delta_rmse_F3_minus_F0": float(np.median(-forcing)),
                "station_win_fraction_F3_rmse_lt_F0": float(np.mean(forcing > 0.0)),
                "descriptive_q25_station_forcing_value_rmse": float(np.quantile(forcing, 0.25)),
                "descriptive_q75_station_forcing_value_rmse": float(np.quantile(forcing, 0.75)),
            }
        )
    document: dict[str, object] = {
        "format": "thermoroute.forcing-v5-observed-point-summary.v1",
        "authority_status": AUTHORITY_STATUS,
        "authority_kind": "POINT_ESTIMATE_ONLY",
        "station_first_estimand": "median_i[RMSE_i(F0)-RMSE_i(F3_full)]",
        "pooled_daily_row_rmse_is_primary": False,
        "difference_of_marginal_medians_is_estimand": False,
        "rows": rows,
        "inference_boundary": {
            "status": INFERENCE_STATUS,
            "inference_authority_published_by_this_builder": False,
            "confidence_intervals_computed": False,
            "p_values_computed": False,
            "bootstrap_computed": False,
            "sign_flip_computed": False,
            "inferential_claims_authorized": False,
            "reserved_separate_path": RESERVED_INFERENCE_DIR.relative_to(ROOT).as_posix(),
        },
    }
    _assert_point_document(document)
    return document


def _assert_point_document(document: Mapping[str, object]) -> None:
    boundary = document.get("inference_boundary")
    if type(boundary) is not dict:
        raise AuthorityError("point document lacks an explicit inference boundary")
    if (
        boundary.get("status") != INFERENCE_STATUS
        or boundary.get("inference_authority_published_by_this_builder") is not False
        or boundary.get("inferential_claims_authorized") is not False
    ):
        raise AuthorityError("point document falsely claims inference authority")
    for key, value in boundary.items():
        if key.endswith("_computed") and value is not False:
            raise AuthorityError("point document contains an inferential computation")


def _independent_raw_example_registry(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in V5.HORIZONS:
        raise AuthorityError("raw-lineage horizon is outside the frozen scope")
    rows: list[pd.DataFrame] = []
    for site, group in raw.groupby("site_id", sort=True):
        ordered = group.sort_values("DATE", kind="mergesort").reset_index(drop=True)
        issue = pd.to_datetime(ordered["DATE"])
        target = issue + pd.to_timedelta(horizon, unit="D")
        y = ordered["WTEMP"].shift(-horizon)
        issue_observed = ordered["WTEMP_observed"].astype(bool)
        target_observed = issue_observed.shift(-horizon, fill_value=False)
        split = np.full(len(ordered), "none", dtype=object)
        for name, (lower_raw, upper_raw) in V5.C.SPLIT.as_dict().items():
            lower, upper = pd.Timestamp(lower_raw), pd.Timestamp(upper_raw)
            mask = issue.between(lower, upper) & target.between(lower, upper)
            split[mask.to_numpy()] = name
        confirm = issue.between(V5.CONFIRM_START, V5.CONFIRM_END) & target.between(
            V5.CONFIRM_START, V5.CONFIRM_END
        )
        split[confirm.to_numpy()] = "confirm"
        rows.append(
            pd.DataFrame(
                {
                    "site_id": np.full(len(ordered), site, dtype=object),
                    "issue_date": issue.to_numpy(copy=True),
                    "target_date": target.to_numpy(copy=True),
                    "y": y.to_numpy(copy=True),
                    "issue_wtemp_observed": issue_observed.to_numpy(copy=True),
                    "target_wtemp_observed": target_observed.to_numpy(copy=True),
                    "split": split,
                    "horizon": np.full(len(ordered), horizon, dtype=np.int16),
                }
            )
        )
    registry = pd.concat(rows, ignore_index=True)
    admissible = (
        registry["issue_wtemp_observed"]
        & registry["target_wtemp_observed"]
        & np.isfinite(registry["y"].to_numpy(dtype=float))
    )
    registry = registry.loc[admissible].copy()
    return registry.sort_values(
        ["split", "site_id", "issue_date"], kind="mergesort"
    ).reset_index(drop=True)


def recompute_observed_lineage_and_raw_cell(
    execution_inputs: Mapping[str, BoundFile],
    reference: pd.DataFrame,
) -> RecomputeEvidence:
    """Rebuild all raw label identities and refit exactly the declared raw cell."""

    v5_bounds = {
        role: V5._BoundFile(
            bound.path,
            bound.payload,
            bound.sha256,
            bound.stat_signature,
        )
        for role, bound in execution_inputs.items()
    }
    try:
        raw, stations, _registry = V5._load_raw_panel_from_bounds(v5_bounds)
    except Exception as exc:
        raise AuthorityError("raw observed panel cannot be rebuilt from sealed inputs") from exc
    raw_frame = raw.frame
    for variable in V5.C.ALL_VARS:
        observed = raw_frame[f"{variable}_observed"].to_numpy(copy=True)
        values = raw_frame[variable].to_numpy(dtype=np.float64, copy=True)
        if observed.dtype != np.dtype("bool") or not np.array_equal(observed, np.isfinite(values)):
            raise AuthorityError("raw observedness is not exact isfinite-before-imputation")
    training_lineage: dict[str, object] = {}
    for horizon in V5.HORIZONS:
        independent = _independent_raw_example_registry(raw_frame, horizon)
        runner_registry = V5.build_raw_example_registry(raw, horizon)
        identity_columns = [
            "site_id",
            "issue_date",
            "target_date",
            "y",
            "issue_wtemp_observed",
            "target_wtemp_observed",
            "split",
            "horizon",
        ]
        if not independent[identity_columns].equals(runner_registry[identity_columns]):
            raise AuthorityError("independent raw label registry differs from runner reconstruction")
        split_records: dict[str, object] = {}
        for split in ("train", "val"):
            rows = independent.loc[independent["split"].eq(split)]
            split_records[split] = {
                "rows": len(rows),
                "key_sha256": V5._canonical_frame_sha256(
                    rows, ("site_id", "issue_date", "target_date")
                ),
                "key_and_y_sha256": V5._canonical_frame_sha256(
                    rows, ("site_id", "issue_date", "target_date", "y")
                ),
                "issue_and_target_raw_observed": True,
            }
        training_lineage[str(horizon)] = split_records

    try:
        V5.C.STATIONS = tuple(stations)
        preprocessing = V5.fit_observed_preprocessing(raw, stations)
        preprocessing_record = V5.preprocessing_content_record(preprocessing)
        imputed = V5.impute_feature_panel(raw, preprocessing.imputer)
        base_tables: dict[int, tuple[pd.DataFrame, tuple[str, ...]]] = {}
        for horizon in V5.HORIZONS:
            base_tables[horizon] = V5.build_observed_feature_table(
                raw, imputed, preprocessing.water_climatology, horizon
            )
        base, columns = base_tables[RECOMPUTED_CELL.horizon]
        arm_table, model_columns = V5.materialize_forcing_features(
            base,
            columns,
            arm=RECOMPUTED_CELL.arm,
            horizon=RECOMPUTED_CELL.horizon,
            raw_future_registry=None,
            meteorology_climatologies=None,
        )
        evaluation = V5.bind_exact_evaluation_rows(
            arm_table,
            reference,
            RECOMPUTED_CELL.horizon,
            model_columns,
        )
        frame, fit = V5.fit_tree_cell(
            arm_table,
            evaluation,
            model_columns,
            preprocessing,
            RECOMPUTED_CELL,
        )
        frame = V5.validate_v5_shard(frame, RECOMPUTED_CELL, reference)
    except Exception as exc:
        raise AuthorityError("declared raw cell cannot be deterministically recomputed") from exc
    raw_lineage = {
        "source_sha256": raw.source_sha256,
        "identity_sha256": raw.identity_sha256,
        "observed_mask_sha256": raw.observed_mask_sha256,
        "full_content_sha256": raw.content_sha256,
        "observed_policy": "exact bool equal to isfinite(raw) before imputation",
        "masks_survived_transform": True,
    }
    imputed_lineage = {
        "raw_source_sha256": imputed.raw_source_sha256,
        "raw_identity_sha256": imputed.raw_identity_sha256,
        "raw_content_sha256": imputed.raw_content_sha256,
        "observed_mask_sha256": imputed.observed_mask_sha256,
        "full_content_sha256": imputed.content_sha256,
    }
    return RecomputeEvidence(
        RECOMPUTED_CELL,
        frame,
        raw_lineage,
        imputed_lineage,
        training_lineage,
        preprocessing_record,
        fit,
    )


def _assert_recomputation_matches(
    snapshot: ResultSnapshot,
    recomputed: RecomputeEvidence,
) -> dict[str, object]:
    manifest = snapshot.manifest
    for field, observed in (
        ("raw_panel_lineage", recomputed.raw_lineage),
        ("imputed_feature_lineage", recomputed.imputed_lineage),
        ("training_identity_by_horizon", recomputed.training_identity_by_horizon),
        ("preprocessing", recomputed.preprocessing),
    ):
        if manifest.get(field) != observed:
            raise AuthorityError(f"runner manifest {field} differs from independent recomputation")
    original = snapshot.frames[recomputed.cell].sort_values(
        ["site_id", "issue_date"], kind="mergesort"
    ).reset_index(drop=True)
    rebuilt = recomputed.frame.sort_values(
        ["site_id", "issue_date"], kind="mergesort"
    ).reset_index(drop=True)
    if tuple(original.columns) != V5.SHARD_COLUMNS or not original.equals(rebuilt):
        raise AuthorityError("raw-cell recomputation differs item-by-item from the formal shard")
    return {
        "cell": _cell_dict(recomputed.cell),
        "rows": len(rebuilt),
        "complete_frame_sha256": V5._canonical_frame_sha256(rebuilt, V5.SHARD_COLUMNS),
        "item_by_item_exact_match": True,
        "raw_inputs_reopened_from_sealed_bindings": True,
        "model_refit_performed": True,
        "fit_evidence_sha256": _canonical_json_sha256(recomputed.fit),
    }


def _git_current_source_authority() -> dict[str, object]:
    root = _git_output("rev-parse", "--show-toplevel").decode().strip()
    if _absolute(Path(root)) != ROOT:
        raise AuthorityError("result-authority Git root differs from the repository root")
    object_format = _git_output("rev-parse", "--show-object-format").decode().strip()
    commit = _git_output("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = _git_output("rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    paths = (BUILDER_PATH, TEST_PATH)
    relative_paths = [path.relative_to(ROOT).as_posix() for path in paths]
    status = _git_output(
        "status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths
    )
    if status:
        raise AuthorityError("result-authority builder/tests are not clean committed Git blobs")
    bindings: dict[str, object] = {}
    for role, path in zip(("builder", "tests"), paths, strict=True):
        bound = _read_stable_regular(path, root=ROOT, label=f"result authority {role}")
        relative = path.relative_to(ROOT).as_posix()
        oid = _git_output("rev-parse", f"HEAD:{relative}").decode().strip()
        payload = _git_output("cat-file", "blob", oid)
        if payload != bound.payload:
            raise AuthorityError("result-authority working source differs from the HEAD blob")
        bindings[role] = {**bound.content_binding, "git_blob_oid": oid}
    return {
        "object_format": object_format,
        "commit": commit,
        "tree": tree,
        "paths": bindings,
        "path_scoped_clean": True,
        "repository_wide_clean_claimed": False,
    }


def _shard_set_binding(shards: Mapping[V5.Cell, BoundFile]) -> dict[str, object]:
    records = [
        {
            "cell": _cell_dict(cell),
            **bound.content_binding,
        }
        for cell, bound in sorted(shards.items())
    ]
    return {
        "count": len(records),
        "files": records,
        "aggregate_sha256": _canonical_json_sha256(records),
    }


def build_candidate(
    *,
    expected_manifest_sha256: str,
    recompute: Callable[
        [Mapping[str, BoundFile], pd.DataFrame], RecomputeEvidence
    ] = recompute_observed_lineage_and_raw_cell,
) -> PointAuthorityCandidate:
    snapshot = capture_result_snapshot(expected_manifest_sha256=expected_manifest_sha256)
    recomputation = recompute(snapshot.execution_inputs, snapshot.reference)
    recompute_record = _assert_recomputation_matches(snapshot, recomputation)
    station = station_metrics_from_shards(snapshot.frames)
    effects = paired_effects_from_station_metrics(station)
    summary = build_point_summary(effects)
    git_source = _git_current_source_authority()
    chain = validate_execution_authority_chain(snapshot.execution_inputs)
    manifest_base: dict[str, object] = {
        "format": AUTHORITY_FORMAT,
        "authority_status": AUTHORITY_STATUS,
        "authority_kind": "POINT_ESTIMATE_ONLY",
        "prospective": False,
        "confirmatory": False,
        "post_outcome_observed_lineage_correction": True,
        "scope": {
            "arms": list(V5.ARMS),
            "models": list(V5.MODELS),
            "horizons": list(V5.HORIZONS),
            "cell_count": 12,
            "cells": [_cell_dict(cell) for cell in expected_cells()],
        },
        "result_bundle": {
            "manifest": snapshot.manifest_bound.content_binding,
            "externally_retained_manifest_sha256_required": True,
            "shards": _shard_set_binding(snapshot.shards),
            "legacy_v4_or_withdrawn_outputs_admitted": False,
        },
        "execution_authority_chain": chain,
        "result_authority_source": git_source,
        "audits": {
            "exact_twelve_cells": True,
            "formal_key_sets_two_sided_exact": True,
            "formal_key_identity_exact": True,
            "formal_y_true_item_by_item_exact": True,
            "cross_cell_y_true_exact": True,
            "cross_cell_damped_anchor_exact": True,
            "raw_observedness_rebuilt_before_imputation": True,
            "training_and_validation_identity_hashes_rebuilt": True,
            "raw_cell_recomputation": recompute_record,
            "station_first_metrics_reconstructed_from_key_level_shards": True,
            "paired_F3_minus_F0_on_identical_station_keys": True,
            "minimum_paired_targets_per_station_lead": MIN_REPORTABLE_KEYS,
            "pooled_daily_row_rmse_used_as_primary": False,
            "difference_of_marginal_station_medians_used_as_effect": False,
        },
        "row_counts": {
            "key_level_rows_across_twelve_shards": sum(
                len(frame) for frame in snapshot.frames.values()
            ),
            "station_metric_rows": len(station),
            "paired_effect_rows": len(effects),
            "summary_rows": len(summary["rows"]),
        },
        "point_output_content_hashes": {
            "station_metrics_canonical_frame_sha256": V5._canonical_frame_sha256(
                station, POINT_STATION_COLUMNS
            ),
            "paired_effects_canonical_frame_sha256": V5._canonical_frame_sha256(
                effects, POINT_EFFECT_COLUMNS
            ),
            "summary_canonical_json_sha256": _canonical_json_sha256(summary),
        },
        "inference_boundary": summary["inference_boundary"],
        "publication": {
            "destination": DEFAULT_AUTHORITY_DIR.relative_to(ROOT).as_posix(),
            "create_only": True,
            "resume": False,
            "overwrite": False,
            "atomic_directory_rename_noreplace": True,
            "manifest_written_last_in_staging": True,
        },
    }
    _assert_point_document(manifest_base)
    return PointAuthorityCandidate(station, effects, summary, manifest_base, snapshot, recomputation)


def _revalidate_candidate_inputs(candidate: PointAuthorityCandidate) -> None:
    _revalidate_bound(candidate.snapshot.manifest_bound, label="precommit result manifest")
    for cell, bound in candidate.snapshot.shards.items():
        _revalidate_bound(bound, label=f"precommit result shard {cell}")
    for role, bound in candidate.snapshot.execution_inputs.items():
        _revalidate_bound(bound, label=f"precommit execution input {role}")
    if _git_current_source_authority() != candidate.manifest_base["result_authority_source"]:
        raise AuthorityError("result-authority Git source changed before publication")


def publish_candidate(
    candidate: PointAuthorityCandidate,
    *,
    output_dir: Path = DEFAULT_AUTHORITY_DIR,
) -> Path:
    if Path(output_dir) != DEFAULT_AUTHORITY_DIR or ".." in Path(output_dir).parts:
        raise AuthorityError(f"point authority destination must be exactly {DEFAULT_AUTHORITY_DIR}")
    if os.path.lexists(DEFAULT_AUTHORITY_DIR):
        raise AuthorityError("point result authority already exists; overwrite/resume is forbidden")
    expected_files: dict[str, dict[str, object]] = {}
    with V5._BundleTransaction(
        DEFAULT_AUTHORITY_DIR,
        allowed_root=V5.FINAL_OUTPUT_ROOT,
        expected_destination_name=DEFAULT_AUTHORITY_DIR.name,
    ) as transaction:
        expected_files[POINT_STATION_FILENAME] = transaction.write_parquet(
            POINT_STATION_FILENAME, candidate.station_metrics
        )
        expected_files[POINT_EFFECT_FILENAME] = transaction.write_parquet(
            POINT_EFFECT_FILENAME, candidate.paired_effects
        )
        expected_files[POINT_SUMMARY_FILENAME] = transaction.write_bytes(
            POINT_SUMMARY_FILENAME, _canonical_json_bytes(candidate.summary)
        )
        manifest = dict(candidate.manifest_base)
        manifest["published_outputs"] = {
            name: dict(binding) for name, binding in sorted(expected_files.items())
        }
        manifest_payload = _canonical_json_bytes(manifest)
        expected_files[POINT_MANIFEST_FILENAME] = transaction.write_bytes(
            POINT_MANIFEST_FILENAME, manifest_payload
        )

        def precommit() -> None:
            _revalidate_candidate_inputs(candidate)
            if _canonical_json_bytes(manifest) != manifest_payload:
                raise AuthorityError("point result-authority manifest changed before commit")

        transaction.commit(expected_files, precommit_check=precommit)
    return DEFAULT_AUTHORITY_DIR


def dry_run_plan() -> dict[str, object]:
    return {
        "format": AUTHORITY_FORMAT,
        "mode": "DRY_RUN_NO_BYTES_READ_NO_MODEL_FIT_NO_WRITE",
        "verification_performed": False,
        "publication_performed": False,
        "result_or_prediction_artifacts_read": False,
        "model_refit_performed": False,
        "result_input": DEFAULT_RESULT_DIR.relative_to(ROOT).as_posix(),
        "result_manifest_pin_required_for_verify_or_publish": True,
        "point_authority_output": DEFAULT_AUTHORITY_DIR.relative_to(ROOT).as_posix(),
        "inference_authority_output": RESERVED_INFERENCE_DIR.relative_to(ROOT).as_posix(),
        "inference_authority_published_by_this_builder": False,
        "scope": {
            "cells": [_cell_dict(cell) for cell in expected_cells()],
            "cell_count": 12,
            "raw_cell_recomputed": _cell_dict(RECOMPUTED_CELL),
        },
        "publication": {
            "default_is_dry_run": True,
            "explicit_publish_flag_required": True,
            "create_only": True,
            "resume": False,
            "overwrite": False,
            "atomic_directory_rename_noreplace": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the byte-free plan (default)")
    mode.add_argument("--verify", action="store_true", help="verify and reconstruct, without writes")
    mode.add_argument("--publish", action="store_true", help="verify then publish create-only")
    parser.add_argument(
        "--expected-result-manifest-sha256",
        help="externally retained SHA256 of the canonical v5 runner manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verify and not args.publish:
        print(json.dumps(dry_run_plan(), sort_keys=True, indent=2))
        return 0
    if args.expected_result_manifest_sha256 is None:
        print(
            "RESULT AUTHORITY REFUSED: --expected-result-manifest-sha256 is required",
            file=sys.stderr,
        )
        return 2
    try:
        candidate = build_candidate(
            expected_manifest_sha256=args.expected_result_manifest_sha256
        )
        if args.publish:
            destination = publish_candidate(candidate)
            result = {
                "verified": True,
                "published": True,
                "authority_kind": "POINT_ESTIMATE_ONLY",
                "destination": destination.relative_to(ROOT).as_posix(),
                "inference_authority_published": False,
            }
        else:
            result = {
                "verified": True,
                "published": False,
                "authority_kind": "POINT_ESTIMATE_ONLY",
                "result_manifest_sha256": candidate.snapshot.manifest_bound.sha256,
                "station_metric_rows": len(candidate.station_metrics),
                "paired_effect_rows": len(candidate.paired_effects),
                "raw_cell_recomputed": _cell_dict(candidate.recomputation.cell),
                "inference_authority_published": False,
            }
    except (AuthorityError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"RESULT AUTHORITY REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
