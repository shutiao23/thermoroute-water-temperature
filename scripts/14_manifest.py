#!/usr/bin/env python3
"""Build and verify the ThermoRoute provenance manifest.

The v1 manifest only hashed result files.  That could prove that a file had not
changed, but not which source, resolved configuration, dependency lock, input
panel, or Git revision produced it.  This v2 manifest records those identities
and an explicit, acyclic parent graph for every scientific artifact.

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

# Scientific inputs and derived evidence.  Logs and retired archives are not
# evidence nodes: they are operational records, not current scientific truth.
ARTIFACT_PATTERNS = (
    "data/*.csv",
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
    "TARGET", "STATIONS", "ALL_VARS", "FORCINGS", "SENTINELS", "LOG1P_VARS",
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
            if not path.is_file() or path in seen:
                continue
            rel_parts = path.relative_to(root).parts
            if "__pycache__" in rel_parts or any(part in {"_archive", "_superseded"}
                                                   for part in rel_parts):
                continue
            if path.name.endswith((".tmp", ".partial")):
                continue
            seen.add(path)
            yield path


def inventory(root: Path, patterns: Iterable[str]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(_iter_files(root, patterns)):
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


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
    candidates = {
        "usgs_predictions": STAGE09_PREDICTIONS_PATH,
        "usgs_panel": "data_usgs/panel_usgs_120v2.parquet",
        "usgs_registry": "data_usgs/station_registry_v1.csv",
        "usgs_scores": STAGE09_SCORES_PATH,
        "legacy_three_site_predictions": "outputs/predictions/predictions.parquet",
        "legacy_three_site_scores": "outputs/tables/scores_all.csv",
    }
    return {key: rel for key, rel in candidates.items() if (root / rel).is_file()}


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
                  git_sha: str) -> dict[str, dict[str, Any]]:
    graph: dict[str, dict[str, Any]] = {
        "@source": {"kind": "source_identity", "sha256": source_sha, "parents": []},
        "@config": {"kind": "resolved_config", "sha256": config_sha, "parents": ["@source"]},
        "@dependencies": {"kind": "dependency_lock", "sha256": dependency_sha,
                          "parents": []},
        "@git": {"kind": "source_revision", "sha256": git_sha, "parents": []},
    }
    input_nodes = sorted(rel for rel in files if _artifact_kind(rel) in {"input_data", "protocol"})
    prediction_nodes = sorted(rel for rel in files if _artifact_kind(rel) == "predictions")
    legacy_three_site_inputs = [
        rel for rel in input_nodes if rel.startswith("data/")
    ]
    usgs_inputs = [rel for rel in input_nodes if rel.startswith("data_usgs/")]

    for rel, meta in sorted(files.items()):
        kind = _artifact_kind(rel)
        parents: list[str]
        if kind in {"input_data", "protocol"}:
            parents = []
        elif kind in {"predictions", "model"}:
            is_legacy_three_site = rel in {
                "outputs/predictions/predictions.parquet",
                "outputs/models/thermoroute_explain.pt",
            }
            data_parents = (
                legacy_three_site_inputs if is_legacy_three_site else usgs_inputs
            )
            parents = ["@git", "@source", "@config", "@dependencies", *data_parents]
        else:
            # Derived summaries can depend on several experiment arms.  Listing all
            # retained prediction nodes is conservative but never understates lineage.
            evidence = prediction_nodes if prediction_nodes else input_nodes
            parents = ["@git", "@source", "@config", "@dependencies", *evidence]
        graph[rel] = {
            "kind": kind,
            "sha256": meta["sha256"],
            "bytes": meta["bytes"],
            "parents": list(dict.fromkeys(parents)),
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
                   source_git_dirty: bool = False) -> dict[str, Any]:
    validate_usgs_current_truth(root)
    source_files = inventory(root, SOURCE_PATTERNS)
    artifact_files = inventory(root, ARTIFACT_PATTERNS)
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
    graph = lineage_graph(root, artifact_files, source_sha, config_sha,
                          dependencies["lock_sha256"], sha256_json(git))
    graph_errors = validate_graph(graph)
    if graph_errors:
        raise RuntimeError("invalid generated lineage graph: " + "; ".join(graph_errors))
    return {
        "schema_version": SCHEMA_VERSION,
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
                    strict_environment: bool = False) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return [f"SCHEMA expected {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"]

    expected_files = manifest.get("files", {})
    actual_files = inventory(root, ARTIFACT_PATTERNS)
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
    for name, rel in truth.items():
        if rel not in expected_files or not (root / rel).is_file():
            errors.append(f"CURRENT_TRUTH_MISSING {name}={rel}")

    graph = manifest.get("dag", {})
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
    args = parser.parse_args()

    root = args.root.resolve()
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
                                 strict_environment=args.strict_environment)
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
