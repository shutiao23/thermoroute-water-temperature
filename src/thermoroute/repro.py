"""Content-addressed experiment and artifact utilities.

The research scripts used to treat a filename such as ``seed0.parquet`` as a
checkpoint.  That permits a file produced by one panel, configuration, or source
tree to be silently reused by another.  This module makes the experiment identity
explicit and validates a sidecar before any cache hit is accepted.

Only stable inputs enter ``run_id``.  The numerical runtime contract (interpreter,
dependency and accelerator versions) is an input because reusing a cache produced
by another numerical stack is not a reproducible cache hit.  Volatile facts such
as timestamp, hostname and duration remain provenance only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


RUN_SCHEMA_VERSION = "thermoroute.run.v1"
ARTIFACT_SCHEMA_VERSION = "thermoroute.artifact.v1"

FORMAL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FORMAL_THREADPOOL_CONTROLLER: Any | None = None
_NATIVE_BINARY_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def configure_deterministic_runtime() -> dict[str, Any]:
    """Apply the formal single-threaded Torch/native-library policy."""
    global _FORMAL_THREADPOOL_CONTROLLER
    for name in FORMAL_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Under ``python -I`` CPython ignores this variable while choosing its hash
    # secret.  It is retained only as a compatibility declaration for
    # non-isolated callers.  The formal contract is deliberately *not* a
    # fixed-hash-secret claim: identity-bearing collections must be
    # canonicalised and sorted before iteration or hashing.
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        from threadpoolctl import threadpool_limits

        _FORMAL_THREADPOOL_CONTROLLER = threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - asserted through native diagnostics
        _FORMAL_THREADPOOL_CONTROLLER = None
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return assert_formal_numerical_policy()


def formal_numerical_policy() -> dict[str, Any]:
    """Return the effective launch policy that can change floating reductions."""
    hash_policy = "canonical-sort-identity-collections-independent-of-hash-secret"
    policy: dict[str, Any] = {
        "thread_environment": {
            name: os.environ.get(name) for name in FORMAL_THREAD_ENVIRONMENT
        },
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_environment_declaration": os.environ.get("PYTHONHASHSEED"),
        "python_hash_randomization_enabled": bool(sys.flags.hash_randomization),
        "python_hash_policy": hash_policy,
        "required": {
            "threads": 1,
            "cublas_workspace_config": ":4096:8",
            "python_hash_policy": hash_policy,
            "torch_deterministic_algorithms": True,
            "tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    try:
        import torch

        policy["torch"] = {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    except ImportError:  # pragma: no cover - Torch is a formal dependency
        policy["torch"] = None
    return policy


def assert_formal_numerical_policy() -> dict[str, Any]:
    """Fail before training when effective thread/determinism knobs drift."""
    policy = formal_numerical_policy()
    if any(
        policy["thread_environment"].get(name) != "1"
        for name in FORMAL_THREAD_ENVIRONMENT
    ):
        raise RuntimeError("formal run requires every BLAS/OpenMP thread count to be 1")
    if policy["cublas_workspace_config"] != ":4096:8":
        raise RuntimeError("formal run requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if policy["python_hash_policy"] != (
        "canonical-sort-identity-collections-independent-of-hash-secret"
    ):
        raise RuntimeError("formal run requires hash-order-independent identities")
    torch_policy = policy.get("torch")
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
    if not isinstance(torch_policy, Mapping) or any(
        torch_policy.get(key) != value for key, value in expected_torch.items()
    ):
        raise RuntimeError("formal Torch numerical policy is not active")
    return policy


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Path):
        return str(value)
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
    return value


def canonical_json(value: Any) -> str:
    """Canonical JSON used for configuration and lineage hashes."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_native_library_identities(
    libraries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one deterministic record per loaded native-library identity.

    ``threadpoolctl`` can report the same shared object more than once when it
    is reached through several import paths.  Handle multiplicity and discovery
    order are not properties of the numerical runtime, so exact duplicate
    identities are folded.  Conversely, different binary bytes, version,
    architecture, API, or threading layer remain different identities and
    therefore change the runtime hash.  Installation paths are deliberately
    excluded: relocating identical binaries must neither change the numerical
    identity nor disclose a builder-specific absolute path.

    Records are ordered lexicographically by their canonical-JSON encoding.
    Launch-time thread counts are deliberately omitted: the effective
    single-thread policy is attested separately by ``formal_numerical_policy``.
    """
    unique: dict[str, dict[str, Any]] = {}
    for library in libraries:
        raw_path = library.get("filepath")
        filepath: Path | None = None
        if raw_path not in (None, ""):
            try:
                candidate = Path(str(raw_path)).expanduser().resolve()
                if candidate.is_file():
                    filepath = candidate
            except (OSError, RuntimeError):
                filepath = None
        binary_sha256: str | None = None
        binary_bytes: int | None = None
        if filepath is not None:
            stat = filepath.stat()
            binary_bytes = int(stat.st_size)
            cache_key = (str(filepath), binary_bytes, int(stat.st_mtime_ns))
            binary_sha256 = _NATIVE_BINARY_HASH_CACHE.get(cache_key)
            if binary_sha256 is None:
                binary_sha256 = sha256_file(filepath)
                _NATIVE_BINARY_HASH_CACHE[cache_key] = binary_sha256
        identity = {
            "user_api": library.get("user_api"),
            "internal_api": library.get("internal_api"),
            "prefix": library.get("prefix"),
            "binary_sha256": binary_sha256,
            "binary_bytes": binary_bytes,
            "version": library.get("version"),
            "architecture": library.get("architecture"),
            "process_abi": {
                "machine": platform.machine(),
                "pointer_bits": int(struct.calcsize("P") * 8),
                "python_cache_tag": getattr(sys.implementation, "cache_tag", None),
            },
            "threading_layer": library.get("threading_layer"),
        }
        unique[canonical_json(identity)] = identity
    return [unique[key] for key in sorted(unique)]


DEFAULT_SOURCE_PATTERNS = (
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


def source_inventory(root: str | Path,
                     patterns: Iterable[str] = DEFAULT_SOURCE_PATTERNS) -> dict[str, str]:
    """Hash the code/config files that define a run, independent of Git state."""
    root = Path(root).resolve()
    files: dict[str, str] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    return dict(sorted(files.items()))


def source_tree_hash(root: str | Path,
                     patterns: Iterable[str] = DEFAULT_SOURCE_PATTERNS) -> str:
    return sha256_json(source_inventory(root, patterns))


def git_state(root: str | Path) -> dict[str, Any]:
    """Best-effort Git provenance; failures are explicit rather than fabricated."""
    root = Path(root)

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    if commit.returncode:
        return {"available": False, "commit": None, "dirty": None}
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
    }


@dataclass(frozen=True)
class RunIdentity:
    """Stable identity of a resolved experiment."""

    run_id: str
    panel_sha256: str
    registry_sha256: str
    config_sha256: str
    source_sha256: str
    runtime_sha256: str
    schema_version: str = RUN_SCHEMA_VERSION

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def resolve_run_identity(*, root: str | Path, panel: str | Path,
                         registry: str | Path, config: Any,
                         source_patterns: Iterable[str] = DEFAULT_SOURCE_PATTERNS
                         ) -> RunIdentity:
    """Resolve a content address from data, registry, configuration, and code."""
    parts = {
        "schema_version": RUN_SCHEMA_VERSION,
        "panel_sha256": sha256_file(panel),
        "registry_sha256": sha256_file(registry),
        "config_sha256": sha256_json(config),
        "source_sha256": source_tree_hash(root, source_patterns),
        "runtime_sha256": sha256_json(numerical_runtime_contract()),
    }
    return RunIdentity(run_id=sha256_json(parts)[:20], **parts)


def numerical_runtime_contract() -> dict[str, Any]:
    """Return stable numerical-runtime facts that participate in cache identity.

    This intentionally excludes hostname, timestamps, thread counts and other
    launch-time knobs.  Those facts are attested separately by
    :func:`environment_fingerprint`.  Exact package and accelerator versions do
    enter the identity: a result produced by another BLAS/ML stack must be
    recomputed instead of being accepted as the same cached run.
    """
    from importlib.metadata import PackageNotFoundError, version

    distributions: dict[str, str | None] = {}
    for distribution in (
        "numpy", "pandas", "scipy", "scikit-learn", "torch", "lightgbm",
        "pyarrow", "statsmodels",
    ):
        try:
            distributions[distribution] = version(distribution)
        except PackageNotFoundError:
            distributions[distribution] = None
    contract: dict[str, Any] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "distributions": distributions,
        "formal_numerical_policy": formal_numerical_policy(),
    }
    try:
        from threadpoolctl import threadpool_info

        native_libraries = _canonical_native_library_identities(
            threadpool_info()
        )
        if not native_libraries or any(
            value.get("binary_sha256") is None for value in native_libraries
        ):
            raise RuntimeError(
                "formal runtime cannot content-bind every loaded native library"
            )
        contract["native_libraries"] = native_libraries
    except ImportError:  # pragma: no cover - dependency is required formally
        contract["native_libraries"] = None
    try:
        import torch

        contract["torch_runtime"] = {
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "mps_built": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
            ),
        }
    except ImportError:  # pragma: no cover - torch is a project dependency
        contract["torch_runtime"] = None
    return contract


def environment_fingerprint() -> dict[str, Any]:
    """Runtime facts for audit logs; these do not enter ``run_id``."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numerical_runtime_contract": numerical_runtime_contract(),
        "numerical_runtime_sha256": sha256_json(numerical_runtime_contract()),
    }
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:  # pragma: no cover - package dependency in real runs
        pass
    try:
        import pandas as pd

        info["pandas"] = pd.__version__
    except ImportError:  # pragma: no cover
        pass
    for distribution, key in (
        ("scipy", "scipy"),
        ("scikit-learn", "scikit_learn"),
        ("lightgbm", "lightgbm"),
        ("pyarrow", "pyarrow"),
        ("statsmodels", "statsmodels"),
    ):
        try:
            from importlib.metadata import version

            info[key] = version(distribution)
        except Exception:  # pragma: no cover - optional diagnostic only
            pass
    try:
        from threadpoolctl import threadpool_info

        info["native_threadpools"] = threadpool_info()
    except Exception:  # pragma: no cover - optional diagnostic only
        pass
    try:
        import torch

        info.update({
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "determinism_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                )
            },
        })
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory": int(properties.total_memory),
                "device_count": int(torch.cuda.device_count()),
            }
    except ImportError:  # pragma: no cover
        pass
    return info


def _fsync_parent_directory(path: Path) -> None:
    """Make a completed rename durable in its containing directory."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        if os.name == "nt":  # Windows has no portable directory fsync.
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Write in the destination directory, fsync, then atomically replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, value: Any) -> None:
    payload = (json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n")
    atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_parquet(frame: Any, path: str | Path, **kwargs: Any) -> None:
    """Atomically write a pandas-compatible frame to Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, **kwargs)
        # Ensure bytes are durable before replacing a previous valid artifact.
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_parent_directory(path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def sidecar_path(artifact: str | Path) -> Path:
    artifact = Path(artifact)
    return artifact.with_name(artifact.name + ".meta.json")


def validate_artifact_sidecar(
    artifact: str | Path,
    *,
    identity: RunIdentity | None = None,
    schema: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Strictly validate an artifact and its complete lineage sidecar."""
    artifact = Path(artifact)
    sidecar = sidecar_path(artifact)
    if not artifact.is_file() or not sidecar.is_file():
        raise ValueError("artifact or lineage sidecar is absent")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact lineage sidecar is invalid JSON") from exc
    expected_keys = {
        "schema_version", "kind", "artifact", "artifact_sha256",
        "artifact_bytes", "content_schema", "run", "parents", "extra",
        "created_utc",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ValueError("artifact lineage sidecar schema is not exact")
    try:
        created = datetime.fromisoformat(str(metadata["created_utc"]))
    except ValueError as exc:
        raise ValueError("artifact lineage timestamp is invalid") from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("artifact lineage timestamp is not timezone-aware")
    if (
        metadata["schema_version"] != ARTIFACT_SCHEMA_VERSION
        or metadata["artifact"] != artifact.name
        or metadata["artifact_bytes"] != artifact.stat().st_size
        or metadata["artifact_sha256"] != sha256_file(artifact)
        or not isinstance(metadata["kind"], str)
        or not metadata["kind"]
        or not isinstance(metadata["parents"], dict)
        or not isinstance(metadata["extra"], dict)
    ):
        raise ValueError("artifact bytes or lineage fields changed")
    parents = metadata["parents"]
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in parents.items()
    ):
        raise ValueError("artifact parent registry is malformed")
    run = metadata["run"]
    run_keys = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "schema_version",
    }
    if (
        not isinstance(run, dict)
        or set(run) != run_keys
        or run.get("schema_version") != RUN_SCHEMA_VERSION
        or not isinstance(run.get("run_id"), str)
        or not run["run_id"]
        or any(
            not isinstance(run.get(field), str) or len(run[field]) != 64
            for field in (
                "panel_sha256", "registry_sha256", "config_sha256",
                "source_sha256", "runtime_sha256",
            )
        )
    ):
        raise ValueError("artifact run identity is malformed")
    if identity is not None and run != identity.as_dict():
        raise ValueError("artifact belongs to another run identity")
    if schema is not None and metadata["content_schema"] != schema:
        raise ValueError("artifact content schema changed")
    if kind is not None and metadata["kind"] != kind:
        raise ValueError("artifact kind changed")
    return metadata


def seal_artifact(artifact: str | Path, identity: RunIdentity, *,
                  kind: str, schema: str | None = None,
                  parents: Mapping[str, str] | None = None,
                  extra: Mapping[str, Any] | None = None) -> Path:
    """Write a validated lineage sidecar for an already completed artifact."""
    artifact = Path(artifact)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    stable = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": kind,
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "content_schema": schema,
        "run": identity.as_dict(),
        "parents": dict(sorted((parents or {}).items())),
        "extra": _jsonable(extra or {}),
    }
    destination = sidecar_path(artifact)
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            existing_stable = dict(existing)
            created = existing_stable.pop("created_utc", None)
            try:
                parsed_created = datetime.fromisoformat(str(created))
            except ValueError:
                parsed_created = None
            if (
                existing_stable == stable
                and parsed_created is not None
                and parsed_created.tzinfo is not None
                and parsed_created.utcoffset() is not None
            ):
                # A sidecar is part of later create-only bundle identities.
                # Preserve its exact bytes when the scientific lineage is
                # unchanged; a wall-clock reseal must not make a retry differ.
                return destination
    metadata = {
        **stable,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(destination, metadata)
    return destination


def cache_is_valid(artifact: str | Path, identity: RunIdentity, *,
                   schema: str | None = None) -> bool:
    """Return true only for an intact artifact produced by the same run."""
    try:
        validate_artifact_sidecar(
            artifact,
            identity=identity,
            schema=schema,
        )
    except (OSError, ValueError):
        return False
    return True


def initialise_run_directory(root: str | Path, identity: RunIdentity, config: Any,
                             *, provenance: Mapping[str, Any] | None = None) -> Path:
    """Create an immutable content-addressed run directory and audit record."""
    run_dir = Path(root) / identity.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run.json"
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "identity": identity.as_dict(),
        "resolved_config": _jsonable(config),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_fingerprint(),
        "git": git_state(Path(root).resolve().parents[1]),
        "provenance": _jsonable(provenance or {}),
    }
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get("identity") != identity.as_dict() or old.get("resolved_config") != _jsonable(config):
            raise RuntimeError(f"run directory collision: {run_dir}")
    else:
        atomic_write_json(metadata_path, payload)
    return run_dir
