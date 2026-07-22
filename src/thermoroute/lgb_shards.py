"""Immutable, fail-closed LightGBM training shards.

Formal Route-A LightGBM training produces one native-text Booster for every
``(run, cohort, seed, horizon, head)`` tuple.  This module makes those members
independently resumable without accepting Python object serialisation or a
best-effort cache hit:

* the shard address is the SHA-256 of its complete input lineage;
* native model text is stored as a content-addressed object;
* both object and shard manifest are published with create-only hard links;
* an existing but invalid shard is an error, never a reason to retrain over it;
* a complete-set manifest is published only after every expected shard loads
  and reproduces its native-text parity prediction exactly.

The cache is an execution aid.  The final model-suite bundle remains the
canonical publication artifact and is built only after ``finalize_shard_set``
returns successfully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .repro import RunIdentity, canonical_json, sha256_file, sha256_json


LIGHTGBM_SHARD_FORMAT = "thermoroute.lightgbm-shard.v1"
LIGHTGBM_SHARD_SET_FORMAT = "thermoroute.lightgbm-shard-set.v1"
LIGHTGBM_HEADS = ("point", "q05", "q50", "q95", "event")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_MANIFEST_KEYS = {
    "format", "lineage", "lineage_sha256", "model", "parity",
}
_LINEAGE_KEYS = {
    "run_id", "run_schema_version", "cohort", "seed", "horizon", "head",
    "source_sha256",
    "runtime_sha256", "panel_sha256", "registry_sha256", "config_sha256",
    "design_key_sha256", "head_config_sha256",
}
_MODEL_KEYS = {"path", "sha256", "bytes"}
_PARITY_KEYS = {
    "rows", "input_sha256", "prediction_sha256", "max_abs_difference", "atol",
}
_SET_KEYS = {
    "format", "run_id", "cohort", "set_sha256", "shard_count", "shards",
}
_SET_ENTRY_KEYS = {
    "lineage_sha256", "manifest_path", "manifest_sha256", "model_sha256",
}
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_MODEL_BYTES = 256 << 20


class LightGBMShardError(RuntimeError):
    """A formal LightGBM shard is incomplete, corrupt, or belongs elsewhere."""


@dataclass(frozen=True)
class LightGBMShardLineage:
    """Exact scientific and numerical identity of one trained Booster."""

    run_id: str
    run_schema_version: str
    cohort: str
    seed: int
    horizon: int
    head: str
    source_sha256: str
    runtime_sha256: str
    panel_sha256: str
    registry_sha256: str
    config_sha256: str
    design_key_sha256: str
    head_config_sha256: str

    @classmethod
    def from_run_identity(
        cls,
        identity: RunIdentity,
        *,
        cohort: str,
        seed: int,
        horizon: int,
        head: str,
        design_key_sha256: str,
        head_config: Mapping[str, Any],
    ) -> "LightGBMShardLineage":
        """Construct a shard lineage from the already-resolved formal run."""
        return cls(
            run_id=identity.run_id,
            run_schema_version=identity.schema_version,
            cohort=str(cohort),
            seed=int(seed),
            horizon=int(horizon),
            head=str(head),
            source_sha256=identity.source_sha256,
            runtime_sha256=identity.runtime_sha256,
            panel_sha256=identity.panel_sha256,
            registry_sha256=identity.registry_sha256,
            config_sha256=identity.config_sha256,
            design_key_sha256=str(design_key_sha256),
            head_config_sha256=sha256_json(head_config),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        _validate_lineage_dict(value)
        return value

    @property
    def sha256(self) -> str:
        return sha256_json(self.as_dict())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _validate_lineage_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LINEAGE_KEYS:
        raise LightGBMShardError("LightGBM shard lineage schema is not exact")
    if (
        not isinstance(value["run_id"], str)
        or not value["run_id"]
        or _SAFE_COMPONENT.fullmatch(value["run_id"]) is None
        or not isinstance(value["run_schema_version"], str)
        or not value["run_schema_version"]
        or _SAFE_COMPONENT.fullmatch(value["run_schema_version"]) is None
        or not isinstance(value["cohort"], str)
        or not value["cohort"]
        or _SAFE_COMPONENT.fullmatch(value["cohort"]) is None
        or type(value["seed"]) is not int
        or value["seed"] < 0
        or type(value["horizon"]) is not int
        or value["horizon"] < 1
        or value["head"] not in LIGHTGBM_HEADS
    ):
        raise LightGBMShardError("LightGBM shard logical identity is malformed")
    for field in (
        "source_sha256", "runtime_sha256", "panel_sha256", "registry_sha256",
        "config_sha256", "design_key_sha256", "head_config_sha256",
    ):
        if not _is_sha256(value[field]):
            raise LightGBMShardError(f"LightGBM shard {field} is malformed")
    return value


def _cache_root(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def shard_manifest_path(
    cache_root: str | Path, lineage: LightGBMShardLineage
) -> Path:
    """Return the deterministic input-addressed manifest path for ``lineage``."""
    root = _cache_root(cache_root)
    value = lineage.as_dict()
    return (
        root / "shards" / value["cohort"] / value["run_id"]
        / f"seed{value['seed']}" / f"h{value['horizon']}"
        / f"{value['head']}-{lineage.sha256}.json"
    )


def _booster(model: Any) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    candidate = getattr(model, "booster_", None)
    if not isinstance(candidate, lgb.Booster):
        raise TypeError("a LightGBM shard must be a fitted native Booster")
    return candidate


def _prediction_digest(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = canonical_json({"dtype": "float64-le", "shape": list(array.shape)})
    digest = hashlib.sha256(header.encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _design_matrix_digest(value: Any) -> str:
    """Hash exact parity inputs, including categorical metadata and row order."""
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    metadata: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        item: dict[str, Any] = {"name": str(column), "dtype": str(series.dtype)}
        if isinstance(series.dtype, pd.CategoricalDtype):
            item["categories"] = [str(category) for category in series.cat.categories]
            item["ordered"] = bool(series.cat.ordered)
        metadata.append(item)
    row_hashes = pd.util.hash_pandas_object(
        frame, index=True, categorize=True
    ).to_numpy(dtype="<u8", copy=True)
    digest = hashlib.sha256(canonical_json({
        "columns": metadata,
        "rows": int(len(frame)),
    }).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def _normalise_key_frame(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    missing = set(columns) - set(frame)
    if missing:
        raise LightGBMShardError(
            f"LightGBM design keys are missing columns: {sorted(missing)}"
        )
    normalised = frame.loc[:, list(columns)].copy()
    for column in columns:
        if column.endswith("date"):
            dates = pd.to_datetime(normalised[column], errors="raise")
            normalised[column] = dates.dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        elif pd.api.types.is_float_dtype(normalised[column]):
            normalised[column] = normalised[column].map(
                lambda item: "NA" if pd.isna(item) else format(float(item), ".17g")
            )
        else:
            normalised[column] = normalised[column].astype(str)
    normalised = normalised.sort_values(list(columns), kind="mergesort")
    return normalised.to_csv(index=False, lineterminator="\n").encode("utf-8")


def lightgbm_design_key_digest(
    partitions: Mapping[str, pd.DataFrame],
    *,
    feature_order: Sequence[str],
    key_columns: Sequence[str] = (
        "site_id", "split", "issue_date", "target_date",
    ),
) -> str:
    """Digest the exact train/validation/evaluation key registries and design order."""
    if not partitions or any(
        not isinstance(name, str) or not name for name in partitions
    ):
        raise LightGBMShardError("LightGBM design partitions are malformed")
    order = [str(column) for column in feature_order]
    if not order or len(order) != len(set(order)):
        raise LightGBMShardError("LightGBM design feature order is malformed")
    digest = hashlib.sha256(canonical_json({
        "feature_order": order,
        "key_columns": [str(column) for column in key_columns],
        "partitions": sorted(partitions),
    }).encode("utf-8"))
    for name in sorted(partitions):
        frame = partitions[name]
        digest.update(b"\0partition\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_normalise_key_frame(frame, key_columns))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_cache_directory(root: Path, directory: Path, *, create: bool) -> bool:
    """Create/check cache descendants one component at a time without symlinks."""
    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise LightGBMShardError("LightGBM shard path escapes its cache root") from exc
    if not root.exists():
        if not create:
            return False
        root.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise LightGBMShardError("LightGBM shard cache root is not a directory")
    current = root
    for component in relative.parts:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create:
                return False
            try:
                current.mkdir()
            except FileExistsError:
                pass
            info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise LightGBMShardError(
                f"LightGBM shard cache contains a non-directory or symlink: {current}"
            )
    return True


def _publish_create_only(root: Path, path: Path, payload: bytes) -> None:
    """Atomically publish bytes without ever replacing an existing pathname."""
    _safe_cache_directory(root, path.parent, create=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            published = True
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise LightGBMShardError(
                    f"refusing to replace a non-identical immutable shard file: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    if not published and not path.is_file():  # defensive against unusual link semantics
        raise LightGBMShardError(f"create-only publication failed: {path}")


def _read_exact_json(path: Path, *, expected_keys: set[str]) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_MANIFEST_BYTES:
        raise LightGBMShardError(f"LightGBM shard manifest is not a bounded regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LightGBMShardError(f"cannot parse LightGBM shard manifest: {path}") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise LightGBMShardError("LightGBM shard manifest schema is not exact")
    return value


def _validate_model_binding(root: Path, value: object) -> Path:
    if not isinstance(value, dict) or set(value) != _MODEL_KEYS:
        raise LightGBMShardError("LightGBM shard model binding schema is not exact")
    if not _is_sha256(value["sha256"]):
        raise LightGBMShardError("LightGBM shard model checksum is malformed")
    expected_relative = f"objects/{value['sha256']}.txt"
    if value["path"] != expected_relative:
        raise LightGBMShardError("LightGBM shard model is not content-addressed")
    if type(value["bytes"]) is not int or not (0 < value["bytes"] <= _MAX_MODEL_BYTES):
        raise LightGBMShardError("LightGBM shard model size is invalid")
    path = root / expected_relative
    if not _safe_cache_directory(root, path.parent, create=False):
        raise LightGBMShardError("LightGBM shard native-text object is missing")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise LightGBMShardError("LightGBM shard native-text object is missing") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size != value["bytes"]
        or sha256_file(path) != value["sha256"]
    ):
        raise LightGBMShardError("LightGBM shard native-text object is corrupt")
    return path


def _validate_parity(value: object, parity_input: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _PARITY_KEYS:
        raise LightGBMShardError("LightGBM shard parity schema is not exact")
    if (
        type(value["rows"]) is not int
        or value["rows"] < 1
        or value["rows"] != len(parity_input)
        or not _is_sha256(value["input_sha256"])
        or value["input_sha256"] != _design_matrix_digest(parity_input)
        or not _is_sha256(value["prediction_sha256"])
        or type(value["max_abs_difference"]) is not float
        or type(value["atol"]) is not float
        or not math.isfinite(value["max_abs_difference"])
        or not math.isfinite(value["atol"])
        or value["atol"] < 0.0
        or value["max_abs_difference"] < 0.0
        or value["max_abs_difference"] > value["atol"]
    ):
        raise LightGBMShardError("LightGBM shard native-text parity record is invalid")
    return value


def try_load_lightgbm_shard(
    cache_root: str | Path,
    *,
    lineage: LightGBMShardLineage,
    parity_input: Any,
) -> lgb.Booster | None:
    """Load a valid shard, return ``None`` only when its manifest is absent.

    A present but stale/corrupt manifest or model raises
    :class:`LightGBMShardError`.  Callers must not silently retrain over it.
    """
    root = _cache_root(cache_root)
    path = shard_manifest_path(root, lineage)
    if not _safe_cache_directory(root, path.parent, create=False):
        return None
    if not path.exists() and not path.is_symlink():
        return None
    manifest = _read_exact_json(path, expected_keys=_MANIFEST_KEYS)
    if manifest["format"] != LIGHTGBM_SHARD_FORMAT:
        raise LightGBMShardError("unsupported LightGBM shard format")
    actual_lineage = _validate_lineage_dict(manifest["lineage"])
    expected_lineage = lineage.as_dict()
    if actual_lineage != expected_lineage:
        raise LightGBMShardError("LightGBM shard belongs to stale lineage")
    if manifest["lineage_sha256"] != lineage.sha256:
        raise LightGBMShardError("LightGBM shard lineage checksum changed")
    model_path = _validate_model_binding(root, manifest["model"])
    parity = _validate_parity(manifest["parity"], parity_input)
    try:
        booster = lgb.Booster(model_file=str(model_path))
        prediction = booster.predict(parity_input, num_threads=1)
    except Exception as exc:  # LightGBM raises several native-wrapper exceptions
        raise LightGBMShardError("LightGBM native-text shard cannot be reconstructed") from exc
    if (
        not np.all(np.isfinite(np.asarray(prediction, dtype=float)))
        or _prediction_digest(prediction) != parity["prediction_sha256"]
    ):
        raise LightGBMShardError("LightGBM shard replay prediction changed")
    return booster


def save_lightgbm_shard(
    cache_root: str | Path,
    *,
    lineage: LightGBMShardLineage,
    model: Any,
    parity_input: Any,
    parity_atol: float = 1e-12,
) -> lgb.Booster:
    """Publish one native-text model shard and return its reconstructed Booster."""
    if not isinstance(parity_atol, float) or not math.isfinite(parity_atol) or parity_atol < 0:
        raise ValueError("LightGBM shard parity_atol must be a finite nonnegative float")
    if len(parity_input) < 1:
        raise ValueError("LightGBM shard parity input is empty")
    root = _cache_root(cache_root)
    existing = try_load_lightgbm_shard(
        root, lineage=lineage, parity_input=parity_input
    )
    candidate = _booster(model)
    text = candidate.model_to_string().encode("utf-8")
    if not text or len(text) > _MAX_MODEL_BYTES:
        raise LightGBMShardError("LightGBM native-text shard has an invalid size")
    model_sha256 = hashlib.sha256(text).hexdigest()
    if existing is not None:
        existing_text = existing.model_to_string().encode("utf-8")
        if hashlib.sha256(existing_text).hexdigest() != model_sha256:
            raise LightGBMShardError(
                "refusing to replace a valid shard with different model content"
            )
        return existing

    try:
        reconstructed = lgb.Booster(model_str=text.decode("utf-8"))
        before = np.asarray(candidate.predict(parity_input, num_threads=1), dtype=float)
        after = np.asarray(reconstructed.predict(parity_input, num_threads=1), dtype=float)
    except Exception as exc:
        raise LightGBMShardError("LightGBM native-text roundtrip failed") from exc
    if (
        before.shape != after.shape
        or not np.all(np.isfinite(before))
        or not np.all(np.isfinite(after))
        or not np.allclose(
            before, after, rtol=0.0, atol=parity_atol
        )
    ):
        difference = math.inf if before.shape != after.shape else float(
            np.max(np.abs(before - after))
        )
        raise LightGBMShardError(
            f"LightGBM native-text roundtrip parity failed: {difference}"
        )
    difference = float(np.max(np.abs(before - after)))
    object_path = root / "objects" / f"{model_sha256}.txt"
    _publish_create_only(root, object_path, text)
    manifest = {
        "format": LIGHTGBM_SHARD_FORMAT,
        "lineage": lineage.as_dict(),
        "lineage_sha256": lineage.sha256,
        "model": {
            "path": f"objects/{model_sha256}.txt",
            "sha256": model_sha256,
            "bytes": len(text),
        },
        "parity": {
            "rows": int(len(parity_input)),
            "input_sha256": _design_matrix_digest(parity_input),
            "prediction_sha256": _prediction_digest(after),
            "max_abs_difference": difference,
            "atol": parity_atol,
        },
    }
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_create_only(root, shard_manifest_path(root, lineage), manifest_payload)
    loaded = try_load_lightgbm_shard(
        root, lineage=lineage, parity_input=parity_input
    )
    if loaded is None:  # pragma: no cover - publication is synchronously visible
        raise LightGBMShardError("published LightGBM shard disappeared")
    return loaded


def _ordered_complete_lineages(
    lineages: Sequence[LightGBMShardLineage],
) -> list[LightGBMShardLineage]:
    if not lineages:
        raise LightGBMShardError("cannot finalize an empty LightGBM shard set")
    ordered = sorted(
        lineages, key=lambda value: (value.seed, value.horizon, LIGHTGBM_HEADS.index(value.head))
    )
    if len({lineage.sha256 for lineage in ordered}) != len(ordered):
        raise LightGBMShardError("LightGBM shard set contains duplicate lineages")
    if len({(lineage.run_id, lineage.cohort) for lineage in ordered}) != 1:
        raise LightGBMShardError("LightGBM shard set mixes runs or cohorts")
    run_fields = {
        (
            lineage.source_sha256, lineage.runtime_sha256, lineage.panel_sha256,
            lineage.registry_sha256, lineage.config_sha256,
            lineage.run_schema_version,
        )
        for lineage in ordered
    }
    if len(run_fields) != 1:
        raise LightGBMShardError("LightGBM shard set mixes formal run identities")
    groups: dict[tuple[int, int], set[str]] = {}
    for lineage in ordered:
        groups.setdefault((lineage.seed, lineage.horizon), set()).add(lineage.head)
    if any(heads != set(LIGHTGBM_HEADS) for heads in groups.values()):
        raise LightGBMShardError("LightGBM shard set is missing a probabilistic head")
    seeds = {lineage.seed for lineage in ordered}
    horizons = {lineage.horizon for lineage in ordered}
    if set(groups) != {(seed, horizon) for seed in seeds for horizon in horizons}:
        raise LightGBMShardError("LightGBM shard set is not a complete seed×horizon product")
    for horizon in horizons:
        if len({
            lineage.design_key_sha256
            for lineage in ordered if lineage.horizon == horizon
        }) != 1:
            raise LightGBMShardError(
                f"LightGBM h{horizon} shard set mixes design-key registries"
            )
    return ordered


def finalize_shard_set(
    cache_root: str | Path,
    *,
    lineages: Sequence[LightGBMShardLineage],
    parity_inputs: Mapping[int, Any],
) -> Path:
    """Validate and create-only publish the exact complete formal shard set."""
    root = _cache_root(cache_root)
    ordered = _ordered_complete_lineages(lineages)
    if set(parity_inputs) != {lineage.horizon for lineage in ordered}:
        raise LightGBMShardError("LightGBM shard-set parity inputs are incomplete")
    entries: list[dict[str, str]] = []
    for lineage in ordered:
        manifest_path = shard_manifest_path(root, lineage)
        loaded = try_load_lightgbm_shard(
            root, lineage=lineage, parity_input=parity_inputs[lineage.horizon]
        )
        if loaded is None:
            raise LightGBMShardError(
                f"LightGBM shard set is incomplete: {lineage.seed}/h{lineage.horizon}/{lineage.head}"
            )
        manifest = _read_exact_json(manifest_path, expected_keys=_MANIFEST_KEYS)
        entries.append({
            "lineage_sha256": lineage.sha256,
            "manifest_path": manifest_path.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "model_sha256": str(manifest["model"]["sha256"]),
        })
    run_id, cohort = ordered[0].run_id, ordered[0].cohort
    set_sha256 = sha256_json(entries)
    document = {
        "format": LIGHTGBM_SHARD_SET_FORMAT,
        "run_id": run_id,
        "cohort": cohort,
        "set_sha256": set_sha256,
        "shard_count": len(entries),
        "shards": entries,
    }
    destination = root / "sets" / cohort / run_id / f"{set_sha256}.json"
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _publish_create_only(root, destination, payload)
    validate_shard_set(destination, cache_root=root, expected_lineages=ordered)
    return destination


def validate_shard_set(
    path: str | Path,
    *,
    cache_root: str | Path,
    expected_lineages: Sequence[LightGBMShardLineage],
) -> dict[str, Any]:
    """Strictly validate a complete-set manifest and all immutable bindings."""
    root = _cache_root(cache_root)
    raw_path = Path(path).expanduser().absolute()
    try:
        raw_path.relative_to(root)
    except ValueError as exc:
        raise LightGBMShardError("LightGBM shard-set manifest escapes cache") from exc
    if not _safe_cache_directory(root, raw_path.parent, create=False):
        raise LightGBMShardError("LightGBM shard-set manifest is missing")
    document = _read_exact_json(raw_path, expected_keys=_SET_KEYS)
    ordered = _ordered_complete_lineages(expected_lineages)
    if (
        document["format"] != LIGHTGBM_SHARD_SET_FORMAT
        or document["run_id"] != ordered[0].run_id
        or document["cohort"] != ordered[0].cohort
        or type(document["shard_count"]) is not int
        or document["shard_count"] != len(ordered)
        or not isinstance(document["shards"], list)
        or len(document["shards"]) != len(ordered)
        or not _is_sha256(document["set_sha256"])
    ):
        raise LightGBMShardError("LightGBM shard-set manifest is malformed")
    expected_hashes = [lineage.sha256 for lineage in ordered]
    actual_hashes: list[str] = []
    for entry in document["shards"]:
        if not isinstance(entry, dict) or set(entry) != _SET_ENTRY_KEYS:
            raise LightGBMShardError("LightGBM shard-set entry schema is not exact")
        if any(not _is_sha256(entry[field]) for field in (
            "lineage_sha256", "manifest_sha256", "model_sha256"
        )):
            raise LightGBMShardError("LightGBM shard-set checksum is malformed")
        relative = Path(str(entry["manifest_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise LightGBMShardError("LightGBM shard-set manifest path escapes cache")
        manifest_path = root / relative
        if not _safe_cache_directory(root, manifest_path.parent, create=False):
            raise LightGBMShardError("LightGBM shard-set manifest binding is missing")
        if not manifest_path.is_file() or sha256_file(manifest_path) != entry["manifest_sha256"]:
            raise LightGBMShardError("LightGBM shard-set manifest binding changed")
        manifest = _read_exact_json(manifest_path, expected_keys=_MANIFEST_KEYS)
        model_path = _validate_model_binding(root, manifest["model"])
        if manifest["lineage_sha256"] != entry["lineage_sha256"]:
            raise LightGBMShardError("LightGBM shard-set lineage binding changed")
        if manifest["model"]["sha256"] != entry["model_sha256"]:
            raise LightGBMShardError("LightGBM shard-set model binding changed")
        if sha256_file(model_path) != entry["model_sha256"]:
            raise LightGBMShardError("LightGBM shard-set native model changed")
        actual_hashes.append(str(entry["lineage_sha256"]))
    if actual_hashes != expected_hashes:
        raise LightGBMShardError("LightGBM shard-set ordering or membership changed")
    if document["set_sha256"] != sha256_json(document["shards"]):
        raise LightGBMShardError("LightGBM shard-set content address changed")
    return document
