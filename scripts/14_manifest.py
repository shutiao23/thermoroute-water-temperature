#!/usr/bin/env python3
"""Build and verify the ThermoRoute provenance manifest.

The v1 manifest only hashed result files.  That could prove that a file had not
changed, but not which source, resolved configuration, dependency lock, input
panel, or Git revision produced it.  This v2 manifest records those identities
and an explicit, acyclic parent graph for its selected top-level artifacts.
Content-bound stage receipts separately close internal member predictions,
checkpoints, and bundles that are not duplicated in this inventory.

Examples
--------
Generate the repository manifest::

    python scripts/14_manifest.py

Verify all bytes, code/config identities, and the lineage graph::

    python scripts/14_manifest.py --check

Generate and verify a clean-room/release manifest without requiring Git::

    python scripts/14_manifest.py --root /path/to/release \
        --manifest /path/to/release/outputs/manifest.json --no-git
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "thermoroute.provenance-manifest.v2"
STAGE09_PREDICTIONS_PATH = (
    "outputs/predictions/usgs_predictions_stage9_v2.parquet"
)
STAGE09_SCORES_PATH = "outputs/tables/usgs_scores.csv"
STAGE09_PRIMARY_MODELS = (
    "Persistence", "DampedPersistence", "Climatology", "LightGBM", "ThermoRoute",
)
_RETIRED_MONITORING_FIGURE_STEMS = (
    "fig1_study_area",
    "fig1_monitoring_site_identifiers",
    "fig2_series_climatology",
    "fig3_results_heatmap",
    "fig4_skill_vs_horizon",
    "fig5_blindtest_trajectory",
    "fig5_development_trajectory",
    "fig6_reliability",
    "fig7_lag_importance",
    "fig7_router_allocation",
    "fig8_dynamic_kappa",
    "fig8_latent_decay_coefficient",
    "fig9_loso",
    "fig9_history_dependent_station_holdout",
    "fig10_flow_lagmaps",
    "fig10_flow_stratified_router",
    "fig11_rev_curves",
)
RETIRED_MONITORING_CASE_OUTPUT_MEMBERS = frozenset({
    "data/processed/panel.parquet",
    "outputs/predictions/predictions.parquet",
    "outputs/tables/scores_all.csv",
    "outputs/models/thermoroute_explain.pt",
    "outputs/tables/explain.npz",
    "outputs/tables/paper_tables.md",
    "outputs/tables/decision_value.csv",
    "outputs/tables/decision_value.md",
    "outputs/reports/data_audit.md",
    "outputs/reports/mechanism_summary.md",
    "outputs/reports/latent_component_diagnostics.md",
    *{
        f"outputs/figures/{stem}.{suffix}"
        for stem in _RETIRED_MONITORING_FIGURE_STEMS
        for suffix in ("png", "pdf")
    },
})
PRE_MODEL_FORBIDDEN_PATHS = (
    "outputs/prelabel/route_a_prelabel_chronology_v1.json",
    "data_usgs/confirmatory_candidate_sites_v1.csv",
    "data_usgs/confirmatory_candidate_sites_v1.provenance.json",
    "data_usgs/raw_snapshots/confirmatory-candidates-v1",
    "data_usgs/confirmatory_site_registry_v1.csv",
    "data_usgs/confirmatory_site_registry_v1.lock.json",
    "data_usgs/confirmatory_actual_inputs_v1.json",
    "outputs/prelabel/route_a_inference_gate_v1.json",
    "data_usgs/raw_snapshots/confirmatory-historical-inputs-v1",
    "data_usgs/raw_snapshots/openmeteo-gfs-previous-runs-v1",
    "data_usgs/confirmatory_predictors",
    "data_usgs/confirmatory_opening_authorization_v1.json",
    "data_usgs/confirmatory",
    "data_usgs/confirmatory_outcomes",
    "outputs/confirmatory",
    "data_usgs/wtemp_daily_max.parquet",
)
DEVELOPMENT_PRELABEL_MANIFEST_ROLE = (
    "DEVELOPMENT_PRELABEL_INVENTORY_NO_CONFIRMATORY_OR_OUTCOME_NAMESPACE"
)
GENERIC_MANIFEST_ROLE = "WORKTREE_OR_STAGED_RELEASE_INVENTORY"
DEVELOPMENT_PRELABEL_MANIFEST_AUTHORITY = (
    "MANIFEST_ALONE_CONFERS_NO_SCIENTIFIC_AUTHORITY; SEPARATE_CONTENT_BOUND_"
    "RECEIPTS_REQUIRED"
)
GENERIC_MANIFEST_AUTHORITY = (
    "BYTE_INVENTORY_ONLY; RELEASE_PROFILE_AND_INDEPENDENT_VALIDATORS_DEFINE_"
    "SCIENTIFIC_AUTHORITY"
)
STAGE09_LINEAGE_INPUT_PATHS = frozenset({
    "data_usgs/panel_usgs_120v2.parquet",
    "data_usgs/station_registry_v1.csv",
})

# Files that can change model behaviour or interpretation.  The manifest itself
# is deliberately excluded to avoid a self-hash cycle.
SOURCE_PATTERNS = (
    "src/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
    "tests/**/*.py",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock*.txt",
    "README.md",
    "paper/**/*.md",
    "paper/**/*.tex",
    "paper/**/*.bib",
    "paper/**/*.bbl",
    "paper/claim_registry*.csv",
)

# Top-level scientific inputs and derived evidence.  Logs, retired archives,
# and internal run members already closed by stage receipts are not duplicated
# here: they are operational records or subordinate nodes, not independent
# current scientific truths.
ARTIFACT_PATTERNS = (
    "data_usgs/panel_usgs*.parquet",
    "data_usgs/confirmatory/**/*.parquet",
    "data_usgs/confirmatory/**/*.csv",
    "data_usgs/confirmatory/**/*.json",
    "data_usgs/station*.csv",
    # Panel freeze specifications and acquisition receipts are JSON inputs too;
    # restricting this to ``station*.json`` silently missed frozen_panel_v1.json.
    "data_usgs/*.json",
    "data_usgs/raw_snapshots/**/*",
    "data_usgs/rejected_sites*.csv",
    "evidence/release_profile_v2.json",
    "protocols/*.md",
    "protocols/*.json",
    "outputs/predictions/**/*.parquet",
    "outputs/predictions/**/*.meta.json",
    "outputs/tables/**/*.csv",
    "outputs/tables/**/*.md",
    "outputs/tables/**/*.npz",
    "outputs/tables/**/*.json",
    "outputs/tables/**/*.parquet",
    "outputs/models/**/*.pt",
    "outputs/models/**/*.json",
    "outputs/models/**/*.txt",
    "outputs/model_replay/**/*.json",
    "outputs/reports/*.md",
    "outputs/reports/*.json",
    "outputs/reports/*.csv",
    "outputs/confirmatory/**/*",
    "outputs/runs/**/run.json",
    "outputs/runs/**/*.meta.json",
    "outputs/figures/*.png",
    "outputs/figures/*.pdf",
    "paper/**/*.pdf",
    "paper/**/*.docx",
)
_DEVELOPMENT_PRELABEL_EXCLUDED_ARTIFACT_PATTERNS = frozenset({
    "data_usgs/confirmatory/**/*.parquet",
    "data_usgs/confirmatory/**/*.csv",
    "data_usgs/confirmatory/**/*.json",
    "data_usgs/*.json",
    "data_usgs/raw_snapshots/**/*",
    "outputs/confirmatory/**/*",
    "paper/**/*.pdf",
    "paper/**/*.docx",
})
DEVELOPMENT_PRELABEL_ARTIFACT_PATTERNS = (
    *(
        pattern
        for pattern in ARTIFACT_PATTERNS
        if pattern not in _DEVELOPMENT_PRELABEL_EXCLUDED_ARTIFACT_PATTERNS
    ),
    "data_usgs/development_*.json",
    "data_usgs/frozen_panel_v1.json",
    "data_usgs/huc_metadata_usgs_v1.provenance.json",
    "data_usgs/rejected_sites*.json",
    "data_usgs/confirmatory_model_suite_v1.json",
    "data_usgs/raw_snapshots/huc-v1/**/*",
    "data_usgs/raw_snapshots/development-predictor-bridge-v1/**/*",
)

RUN_SOURCE_PATTERNS = (
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

DIRECT_DISTRIBUTIONS = (
    "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch",
    "matplotlib", "statsmodels", "pyarrow", "pytest", "dataretrieval",
)

CONFIG_NAMES = (
    "TARGET", "ALL_VARS", "FORCINGS", "SENTINELS", "LOG1P_VARS",
    "HORIZONS", "QUANTILES", "EXCEEDANCE_QUANTILE", "SPLIT", "FEATURE_SETS",
    "SHORT_LAGS", "ROLLING_WINDOWS", "CONTEXT_LENGTH", "MAX_ROUTER_LAG",
    "SEASONAL_HARMONICS", "SEEDS", "PRIMARY_SEED", "SEASONAL_PERIOD", "TRAIN",
    "DELTA_SCALE", "USGS_SEEDS",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_jsonable(value), indent=2, sort_keys=True,
                          ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _iter_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen:
                continue
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise RuntimeError(
                    f"MANIFEST_UNSAFE_ARTIFACT: cannot lstat {path}"
                ) from exc
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    "MANIFEST_UNSAFE_ARTIFACT: matched path is not a regular file: "
                    f"{path}"
                )
            rel_parts = path.relative_to(root).parts
            if "__pycache__" in rel_parts or any(part in {"_archive", "_superseded"}
                                                   for part in rel_parts):
                continue
            if path.name.endswith((".tmp", ".partial")):
                continue
            seen.add(path)
            yield path


def _inventory_file_binding(root: Path, path: Path) -> dict[str, Any]:
    """Hash one single-link file through no-follow directory descriptors.

    The development manifest is a byte inventory, not an authority receipt, but
    even an inventory must not follow a renamed symlink or hardlink into a
    withdrawn/outcome namespace.  Descriptor-relative traversal also prevents a
    parent-directory symlink from escaping the lexical repository root.
    """
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise RuntimeError(
            f"MANIFEST_UNSAFE_ARTIFACT: path escapes root: {lexical_path}"
        ) from exc
    if not relative.parts:
        raise RuntimeError("MANIFEST_UNSAFE_ARTIFACT: repository root is not a file")

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "MANIFEST_UNSAFE_ARTIFACT: platform lacks no-follow descriptor support"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    file_flags |= os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)

    descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        root_descriptor = os.open(lexical_root, directory_flags)
        descriptors.append(root_descriptor)
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise RuntimeError(
                "MANIFEST_UNSAFE_ARTIFACT: repository root is not a directory"
            )

        parent_descriptor = root_descriptor
        for component in relative.parts[:-1]:
            try:
                child_descriptor = os.open(
                    component, directory_flags, dir_fd=parent_descriptor
                )
            except OSError as exc:
                raise RuntimeError(
                    "MANIFEST_UNSAFE_ARTIFACT: path contains an unsafe directory "
                    f"component: {relative.as_posix()}"
                ) from exc
            child_metadata = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_metadata.st_mode):
                os.close(child_descriptor)
                raise RuntimeError(
                    "MANIFEST_UNSAFE_ARTIFACT: path crosses a non-directory: "
                    f"{relative.as_posix()}"
                )
            descriptors.append(child_descriptor)
            parent_descriptor = child_descriptor

        name = relative.parts[-1]
        try:
            file_descriptor = os.open(name, file_flags, dir_fd=parent_descriptor)
        except OSError as exc:
            raise RuntimeError(
                "MANIFEST_UNSAFE_ARTIFACT: cannot open file without following links: "
                f"{relative.as_posix()}"
            ) from exc
        before = os.fstat(file_descriptor)
        linked_before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
        )
        linked_identity_before = (
            linked_before.st_dev,
            linked_before.st_ino,
            linked_before.st_mode,
            linked_before.st_nlink,
            linked_before.st_size,
            linked_before.st_mtime_ns,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or linked_identity_before != identity_before
        ):
            raise RuntimeError(
                "MANIFEST_UNSAFE_ARTIFACT: path is not a single-link regular file: "
                f"{relative.as_posix()}"
            )

        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(file_descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)

        after = os.fstat(file_descriptor)
        linked_after = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
        )
        linked_identity_after = (
            linked_after.st_dev,
            linked_after.st_ino,
            linked_after.st_mode,
            linked_after.st_nlink,
            linked_after.st_size,
            linked_after.st_mtime_ns,
        )
        if (
            identity_after != identity_before
            or linked_identity_after != identity_before
            or byte_count != before.st_size
        ):
            raise RuntimeError(
                "MANIFEST_UNSAFE_ARTIFACT: file changed while being inventoried: "
                f"{relative.as_posix()}"
            )
        return {"sha256": digest.hexdigest(), "bytes": byte_count}
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def inventory(root: Path, patterns: Iterable[str]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(_iter_files(root, patterns)):
        rel = path.relative_to(root).as_posix()
        files[rel] = _inventory_file_binding(root, path)
    return files


def retired_monitoring_case_outputs(root: Path) -> tuple[str, ...]:
    """Return retired-case products that would contaminate Route-A evidence."""
    return tuple(
        relative
        for relative in sorted(RETIRED_MONITORING_CASE_OUTPUT_MEMBERS)
        if (root / relative).exists() or (root / relative).is_symlink()
    )


def assert_route_a_artifact_boundary(root: Path) -> None:
    present = retired_monitoring_case_outputs(root)
    if present:
        raise RuntimeError(
            "ROUTE_A_RETIRED_MONITORING_CASE_ARTIFACT_PRESENT: "
            + ", ".join(present)
        )


def pre_model_forbidden_paths(root: Path) -> tuple[str, ...]:
    """Use metadata only to find paths forbidden before the model freeze."""
    return tuple(
        relative
        for relative in PRE_MODEL_FORBIDDEN_PATHS
        if os.path.lexists(root / relative)
    )


def assert_development_prelabel_boundary(root: Path) -> None:
    present = pre_model_forbidden_paths(root)
    if present:
        raise RuntimeError(
            "ROUTE_A_PREMODEL_FORBIDDEN_PATH_PRESENT_WITHOUT_READING: "
            + ", ".join(present)
        )


def resolved_config(root: Path) -> dict[str, Any]:
    """Load public experiment constants without importing the package __init__."""
    path = root / "src" / "thermoroute" / "config.py"
    if not path.is_file():
        return {"config_file": None}
    name = "_thermoroute_manifest_config"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load configuration from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return {key: _jsonable(getattr(module, key)) for key in CONFIG_NAMES
                if hasattr(module, key)}
    finally:
        sys.modules.pop(name, None)


def dependency_identity(root: Path) -> dict[str, Any]:
    lock_files = {}
    for name in ("requirements-lock.txt", "requirements.txt", "pyproject.toml"):
        path = root / name
        if path.is_file():
            lock_files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    installed = {}
    for distribution in DIRECT_DISTRIBUTIONS:
        try:
            installed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed[distribution] = None
    return {
        "lock_files": lock_files,
        "lock_sha256": sha256_json(lock_files),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "installed_direct": installed,
    }


def git_state(root: Path, disabled: bool = False) -> dict[str, Any]:
    if disabled:
        return {"available": False, "reason": "disabled"}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=root, text=True,
                              capture_output=True, check=False)

    top = run("rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        return {"available": False, "reason": "not a git worktree root"}
    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    status = run("status", "--porcelain", "--untracked-files=all")
    dirty_paths = [line[3:] for line in status.stdout.splitlines() if len(line) >= 4]
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "tree": tree.stdout.strip(),
        "dirty": bool(dirty_paths),
        "dirty_paths": sorted(dirty_paths),
    }


def _artifact_kind(rel: str) -> str:
    if rel.startswith(("data/", "data_usgs/")):
        return "input_data"
    if rel.startswith("protocols/"):
        return "protocol"
    if rel.startswith("outputs/predictions/"):
        return "predictions"
    if rel.startswith("outputs/models/"):
        return "model"
    if rel.startswith("outputs/tables/"):
        return "table"
    if rel.startswith("outputs/reports/"):
        return "report"
    if rel.startswith("outputs/figures/"):
        return "figure"
    return "artifact"


def _current_truth(root: Path) -> dict[str, str]:
    # This document is only a broad byte inventory.  Canonical-input, model,
    # replay, and release receipts independently establish scientific
    # authority, so filename presence must never manufacture "current truth".
    del root
    return {}


def _run_source_sha256(root: Path) -> str:
    # Run identity deliberately uses the same inclusion semantics as
    # thermoroute.repro, chronology, the isolated opening contract, and the
    # release verifier.  The broader manifest inventory may omit retired
    # evidence directories, but protected source bytes must not diverge here.
    files: dict[str, str] = {}
    for pattern in RUN_SOURCE_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    return sha256_json(dict(sorted(files.items())))


def validate_usgs_current_truth(root: Path) -> None:
    """Reject a stale or mixed-generation Stage-9 canonical artifact pair."""
    prediction = root / STAGE09_PREDICTIONS_PATH
    scores = root / STAGE09_SCORES_PATH
    if not prediction.is_file() and not scores.is_file():
        return
    if not prediction.is_file() or not scores.is_file():
        raise RuntimeError(
            "USGS_CURRENT_TRUTH_STALE: canonical Stage-9 predictions/scores pair "
            "is incomplete"
        )
    sidecar = prediction.with_name(prediction.name + ".meta.json")
    if not sidecar.is_file():
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: prediction lineage sidecar is missing")
    try:
        lineage = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: invalid lineage JSON") from exc
    expected = {
        "schema_version": "thermoroute.artifact.v1",
        "kind": "canonical_stage9_usgs_predictions",
        "content_schema": "thermoroute.predictions.v1",
        "artifact_sha256": sha256_file(prediction),
        "artifact_bytes": prediction.stat().st_size,
    }
    wrong = {key: (lineage.get(key), value) for key, value in expected.items()
             if lineage.get(key) != value}
    if wrong:
        raise RuntimeError(f"USGS_CURRENT_TRUTH_STALE: lineage mismatch {wrong}")
    run = lineage.get("run", {})
    panel = root / "data_usgs" / "panel_usgs_120v2.parquet"
    registry_path = root / "data_usgs" / "station_registry_v1.csv"
    if (
        run.get("panel_sha256") != sha256_file(panel)
        or run.get("registry_sha256") != sha256_file(registry_path)
        or run.get("source_sha256") != _run_source_sha256(root)
    ):
        raise RuntimeError(
            "USGS_CURRENT_TRUTH_STALE: panel/registry/source identity changed"
        )
    if not isinstance(lineage.get("parents"), dict) or not lineage["parents"]:
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: immutable parent lineage is absent")

    import pandas as pd

    frame = pd.read_parquet(
        prediction,
        columns=[
            "model", "site_id", "horizon", "split", "issue_date",
            "target_date", "y_true",
        ],
    )
    registry = pd.read_csv(registry_path, dtype={"site_no": "string"})
    stable = set(registry.site_no.astype(str).str.strip())
    actual = set(frame.site_id.astype(str).str.strip())
    if not actual or not actual <= stable:
        raise RuntimeError(
            "USGS_CURRENT_TRUTH_STALE: legacy or unknown station identifiers"
        )
    primary = STAGE09_PRIMARY_MODELS
    test = frame[frame.split.eq("test") & frame.model.isin(primary)].copy()
    if set(test.model.astype(str)) != set(primary):
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: a primary model is absent")
    key = ["site_id", "horizon", "issue_date", "target_date"]
    registries = {
        model: set(group[key].itertuples(index=False, name=None))
        for model, group in test.groupby("model")
    }
    first = registries[primary[0]]
    if not first or any(registries[model] != first for model in primary):
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: primary forecast keys differ")
    if not _truth_matches_at_model_precision(test, key=key):
        raise RuntimeError("USGS_CURRENT_TRUTH_STALE: models disagree on y_true")


def _truth_matches_at_model_precision(
    frame: Any, *, key: list[str]
) -> bool:
    """Use the exact float32 label semantics of the neural window registry."""
    import numpy as np
    import pandas as pd

    truth = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
    converted = truth.astype(np.float32)
    if not np.isfinite(truth).all() or not np.isfinite(converted).all():
        return False
    audit = frame.loc[:, key].copy()
    audit["_truth_float32"] = converted
    return bool(audit.groupby(key, dropna=False)["_truth_float32"].nunique().le(1).all())


def lineage_graph(root: Path, files: Mapping[str, Mapping[str, Any]],
                  source_sha: str, config_sha: str, dependency_sha: str,
                  git_sha: str, *,
                  development_prelabel: bool = False) -> dict[str, dict[str, Any]]:
    graph: dict[str, dict[str, Any]] = {
        "@source": {"kind": "source_identity", "sha256": source_sha, "parents": []},
        "@config": {"kind": "resolved_config", "sha256": config_sha, "parents": ["@source"]},
        "@dependencies": {"kind": "dependency_lock", "sha256": dependency_sha,
                          "parents": []},
        "@git": {"kind": "source_revision", "sha256": git_sha, "parents": []},
    }
    input_nodes = sorted(rel for rel in files if _artifact_kind(rel) in {"input_data", "protocol"})
    prediction_nodes = sorted(rel for rel in files if _artifact_kind(rel) == "predictions")
    usgs_inputs = [rel for rel in input_nodes if rel in STAGE09_LINEAGE_INPUT_PATHS]

    for rel, meta in sorted(files.items()):
        kind = _artifact_kind(rel)
        node_kind = kind
        parents: list[str]
        authority = "CONTENT_HASH_INVENTORY"
        if kind in {"input_data", "protocol"}:
            parents = []
        elif development_prelabel and rel.startswith("outputs/"):
            node_kind = "workspace_inventory"
            authority = (
                "WORKSPACE_INVENTORY_ONLY_REQUIRES_SEPARATE_RECEIPT_VALIDATION"
            )
            parents = ["@git", "@source", "@config", "@dependencies"]
        elif kind in {"predictions", "model"}:
            parents = ["@git", "@source", "@config", "@dependencies", *usgs_inputs]
        else:
            # Derived summaries can depend on several experiment arms.  Listing all
            # retained prediction nodes is conservative but never understates lineage.
            evidence = prediction_nodes if prediction_nodes else input_nodes
            parents = ["@git", "@source", "@config", "@dependencies", *evidence]
        graph[rel] = {
            "kind": node_kind,
            "sha256": meta["sha256"],
            "bytes": meta["bytes"],
            "parents": list(dict.fromkeys(parents)),
            "authority": authority,
        }
    return graph


def validate_graph(graph: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for node, meta in graph.items():
        for parent in meta.get("parents", []):
            if parent not in graph:
                errors.append(f"DAG_UNKNOWN_PARENT {node} -> {parent}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            errors.append(f"DAG_CYCLE {node}")
            return
        visiting.add(node)
        for parent in graph.get(node, {}).get("parents", []):
            if parent in graph:
                visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return errors


def supplied_git_state(commit: str, tree: str, *, dirty: bool) -> dict[str, Any]:
    """Validate revision metadata supplied while staging a Git-less release."""
    object_id = re.compile(r"^[0-9a-fA-F]{40}([0-9a-fA-F]{24})?$")
    if not object_id.fullmatch(commit) or not object_id.fullmatch(tree):
        raise ValueError("supplied Git commit/tree must be 40- or 64-digit object IDs")
    return {
        "available": True,
        "commit": commit.lower(),
        "tree": tree.lower(),
        "dirty": bool(dirty),
        "dirty_paths": [],
        "source": "release-builder",
    }


def build_manifest(root: Path, *, no_git: bool = False,
                   source_git_commit: str | None = None,
                   source_git_tree: str | None = None,
                   source_git_dirty: bool = False,
                   development_prelabel: bool = False) -> dict[str, Any]:
    assert_route_a_artifact_boundary(root)
    if development_prelabel:
        assert_development_prelabel_boundary(root)
    source_files = inventory(root, SOURCE_PATTERNS)
    artifact_patterns = (
        DEVELOPMENT_PRELABEL_ARTIFACT_PATTERNS
        if development_prelabel
        else ARTIFACT_PATTERNS
    )
    artifact_files = inventory(root, artifact_patterns)
    if development_prelabel:
        # The development patterns cannot open a forbidden namespace.  This
        # second metadata-only check detects creation during inventory without
        # ever hashing the new path.
        assert_development_prelabel_boundary(root)
    config = resolved_config(root)
    dependencies = dependency_identity(root)
    source_sha = sha256_json(source_files)
    config_sha = sha256_json(config)
    if (source_git_commit is None) != (source_git_tree is None):
        raise ValueError("source Git commit and tree must be supplied together")
    git = (supplied_git_state(source_git_commit, source_git_tree,
                              dirty=source_git_dirty)
           if source_git_commit is not None
           else git_state(root, disabled=no_git))
    graph = lineage_graph(
        root,
        artifact_files,
        source_sha,
        config_sha,
        dependencies["lock_sha256"],
        sha256_json(git),
        development_prelabel=development_prelabel,
    )
    graph_errors = validate_graph(graph)
    if graph_errors:
        raise RuntimeError("invalid generated lineage graph: " + "; ".join(graph_errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_role": (
            DEVELOPMENT_PRELABEL_MANIFEST_ROLE
            if development_prelabel
            else GENERIC_MANIFEST_ROLE
        ),
        "scientific_evidence_authority": (
            DEVELOPMENT_PRELABEL_MANIFEST_AUTHORITY
            if development_prelabel
            else GENERIC_MANIFEST_AUTHORITY
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root_name": root.name,
        "git": git,
        "source": {"sha256": source_sha, "files": source_files},
        "resolved_config": {"sha256": config_sha, "values": config},
        "dependencies": dependencies,
        "current_truth": _current_truth(root),
        "n_files": len(artifact_files),
        "files": artifact_files,
        "dag": graph,
    }


def verify_manifest(root: Path, manifest: Mapping[str, Any], *,
                    no_git: bool = False, strict_git: bool = False,
                    strict_environment: bool = False,
                    expected_role: str | None = None) -> list[str]:
    errors: list[str] = []
    for relative in retired_monitoring_case_outputs(root):
        errors.append(f"RETIRED_MONITORING_CASE_ARTIFACT_PRESENT {relative}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return [f"SCHEMA expected {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"]
    role = manifest.get("manifest_role", GENERIC_MANIFEST_ROLE)
    if role not in {DEVELOPMENT_PRELABEL_MANIFEST_ROLE, GENERIC_MANIFEST_ROLE}:
        errors.append(f"MANIFEST_ROLE_UNKNOWN {role!r}")
    if expected_role is not None and role != expected_role:
        errors.append(f"MANIFEST_ROLE_EXPECTED {expected_role!r}, got {role!r}")
    if role == DEVELOPMENT_PRELABEL_MANIFEST_ROLE:
        for relative in pre_model_forbidden_paths(root):
            errors.append(f"PREMODEL_FORBIDDEN_PATH_PRESENT {relative}")
    expected_authority = (
        DEVELOPMENT_PRELABEL_MANIFEST_AUTHORITY
        if role == DEVELOPMENT_PRELABEL_MANIFEST_ROLE
        else GENERIC_MANIFEST_AUTHORITY
    )
    if manifest.get("scientific_evidence_authority") != expected_authority:
        errors.append("SCIENTIFIC_EVIDENCE_AUTHORITY_CHANGED")

    expected_files = manifest.get("files", {})
    artifact_patterns = (
        DEVELOPMENT_PRELABEL_ARTIFACT_PATTERNS
        if role == DEVELOPMENT_PRELABEL_MANIFEST_ROLE
        else ARTIFACT_PATTERNS
    )
    actual_files = inventory(root, artifact_patterns)
    if role == DEVELOPMENT_PRELABEL_MANIFEST_ROLE:
        for relative in pre_model_forbidden_paths(root):
            marker = f"PREMODEL_FORBIDDEN_PATH_PRESENT {relative}"
            if marker not in errors:
                errors.append(marker)
    for rel, expected in expected_files.items():
        actual = actual_files.get(rel)
        if actual is None:
            errors.append(f"MISSING {rel}")
        elif actual != expected:
            errors.append(f"CHANGED {rel}")
    for rel in sorted(set(actual_files) - set(expected_files)):
        errors.append(f"UNTRACKED_ARTIFACT {rel}")

    actual_source = inventory(root, SOURCE_PATTERNS)
    source = manifest.get("source", {})
    if source.get("files") != actual_source or source.get("sha256") != sha256_json(actual_source):
        errors.append("SOURCE_IDENTITY_CHANGED")

    actual_config = resolved_config(root)
    config = manifest.get("resolved_config", {})
    if config.get("values") != actual_config or config.get("sha256") != sha256_json(actual_config):
        errors.append("RESOLVED_CONFIG_CHANGED")

    actual_dependencies = dependency_identity(root)
    dependencies = manifest.get("dependencies", {})
    if dependencies.get("lock_files") != actual_dependencies["lock_files"] or \
            dependencies.get("lock_sha256") != actual_dependencies["lock_sha256"]:
        errors.append("DEPENDENCY_LOCK_CHANGED")
    if strict_environment and (
            dependencies.get("runtime") != actual_dependencies["runtime"] or
            dependencies.get("installed_direct") != actual_dependencies["installed_direct"]):
        errors.append("RUNTIME_ENVIRONMENT_CHANGED")

    truth = manifest.get("current_truth", {})
    if truth != _current_truth(root):
        errors.append("CURRENT_TRUTH_CHANGED")
    for name, rel in truth.items():
        if rel not in expected_files or not (root / rel).is_file():
            errors.append(f"CURRENT_TRUTH_MISSING {name}={rel}")

    graph = manifest.get("dag", {})
    expected_graph = lineage_graph(
        root,
        actual_files,
        sha256_json(actual_source),
        sha256_json(actual_config),
        actual_dependencies["lock_sha256"],
        sha256_json(manifest.get("git", {})),
        development_prelabel=(role == DEVELOPMENT_PRELABEL_MANIFEST_ROLE),
    )
    if graph != expected_graph:
        errors.append("DAG_CONTENT_OR_AUTHORITY_CHANGED")
    errors.extend(validate_graph(graph))
    for rel, meta in expected_files.items():
        node = graph.get(rel)
        if node is None:
            errors.append(f"DAG_NODE_MISSING {rel}")
        elif node.get("sha256") != meta.get("sha256"):
            errors.append(f"DAG_HASH_MISMATCH {rel}")
    for virtual, expected_sha in (("@source", source.get("sha256")),
                                  ("@config", config.get("sha256")),
                                  ("@dependencies", dependencies.get("lock_sha256")),
                                  ("@git", sha256_json(manifest.get("git", {})))):
        if graph.get(virtual, {}).get("sha256") != expected_sha:
            errors.append(f"DAG_VIRTUAL_HASH_MISMATCH {virtual}")

    if strict_git and not no_git:
        expected_git = manifest.get("git", {})
        actual_git = git_state(root)
        if not expected_git.get("available") or not actual_git.get("available"):
            errors.append("GIT_STATE_UNAVAILABLE")
        else:
            for key in ("commit", "tree"):
                if expected_git.get(key) != actual_git.get(key):
                    errors.append(f"GIT_{key.upper()}_CHANGED")
            if actual_git.get("dirty"):
                errors.append("GIT_WORKTREE_DIRTY")
    return errors


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root,
                        help="repository or extracted-release root")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest path (default: ROOT/outputs/manifest.json)")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    parser.add_argument("--no-git", action="store_true",
                        help="do not require or record a Git worktree")
    parser.add_argument("--strict-git", action="store_true",
                        help="also require the recorded clean commit/tree")
    parser.add_argument("--strict-environment", action="store_true",
                        help="also require exact runtime and installed direct versions")
    parser.add_argument("--source-git-commit", default=None,
                        help="origin commit to bind into a staged Git-less release")
    parser.add_argument("--source-git-tree", default=None,
                        help="origin tree to bind into a staged Git-less release")
    parser.add_argument("--source-git-dirty", action="store_true",
                        help="mark supplied release-builder revision as dirty")
    parser.add_argument(
        "--check-route-a-boundary",
        action="store_true",
        help="metadata-only pre-model check for retired or outcome paths",
    )
    parser.add_argument(
        "--development-prelabel",
        action="store_true",
        help=(
            "build or check the development-prelabel role only while "
            "confirmatory/outcome paths are absent"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if args.check_route_a_boundary:
        try:
            assert_route_a_artifact_boundary(root)
            assert_development_prelabel_boundary(root)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("Route-A artifact boundary OK")
        return 0
    manifest_path = (args.manifest.resolve() if args.manifest else
                     root / "outputs" / "manifest.json")
    if args.check:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"MANIFEST_UNREADABLE {manifest_path}: {exc}", file=sys.stderr)
            return 2
        errors = verify_manifest(root, manifest, no_git=args.no_git,
                                 strict_git=args.strict_git,
                                 strict_environment=args.strict_environment,
                                 expected_role=(
                                     DEVELOPMENT_PRELABEL_MANIFEST_ROLE
                                     if args.development_prelabel else None
                                 ))
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"manifest OK: {manifest.get('n_files', 0)} artifacts, "
              f"source {manifest['source']['sha256'][:12]}, DAG {len(manifest['dag'])} nodes")
        return 0

    manifest = build_manifest(
        root,
        no_git=args.no_git,
        source_git_commit=args.source_git_commit,
        source_git_tree=args.source_git_tree,
        source_git_dirty=args.source_git_dirty,
        development_prelabel=args.development_prelabel,
    )
    atomic_write_json(manifest_path, manifest)
    print(f"wrote {manifest_path}: {manifest['n_files']} artifacts, "
          f"source {manifest['source']['sha256'][:12]}, DAG {len(manifest['dag'])} nodes")
    if manifest["git"].get("dirty"):
        print(f"warning: Git worktree is dirty ({len(manifest['git']['dirty_paths'])} paths)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
