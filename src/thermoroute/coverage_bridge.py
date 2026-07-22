"""Physical, fail-closed bridge for the Route-A temporal coverage audit.

The pure coverage core deliberately knows nothing about repository paths.  This
module supplies the missing physical evidence boundary: it derives all eleven
logical source names from the frozen authorization/state, opens every physical
file without following links, hashes and parses it through the same descriptor,
and then replays the deterministic core.  Producer-owned in-memory frames are
never accepted.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .coverage_audit import (
    AUDIT_FORMAT,
    AVAILABILITY_COLUMNS,
    COMPARISON_MODELS,
    FORECAST_KEY_DIGEST_DOMAIN,
    HORIZONS,
    MODEL_REGISTRY,
    POLICY_FILE_SHA256,
    POLICY_RELATIVE,
    PREDICTION_COLUMNS,
    SOURCE_BINDING_KEYS,
    TARGET_END,
    TARGET_START,
    Y_TRUE_DIGEST_DOMAIN,
    build_temporal_coverage_audit,
    validate_temporal_coverage_audit,
    validate_temporal_coverage_policy_bytes,
)
from .repro import canonical_json


CONSTRUCTION_BUFFER_START = "2020-11-30"
CORE_PROJECTION_START = "2020-12-01"
PARQUET_BATCH_ROWS = 8192
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TRUSTED_SOURCE_STATE_KEYS = {
    "temporal_full_predictions": "temporal_predictions",
    "external_full_predictions": "external_predictions",
    "availability_registry": "availability_registry",
    "statistics": "statistics",
}
_SOURCE_STATE_KEYS = {
    "acquisition_manifest": "acquisition_manifest",
    "temporal_normalized_outcomes": "temporal_outcomes",
    "external_normalized_outcomes": "external_outcomes",
    **_TRUSTED_SOURCE_STATE_KEYS,
}
_RECEIPT_SOURCE_KEYS = {
    "acquisition_manifest": "acquisition_manifest",
    "temporal_normalized_outcomes": "temporal_normalized_outcomes",
    "external_normalized_outcomes": "external_normalized_outcomes",
    "temporal_full_predictions": "temporal_predictions",
    "external_full_predictions": "external_predictions",
    "availability_registry": "availability_registry",
    "statistics": "statistics",
}


class CoverageBridgeError(RuntimeError):
    """A canonical path, physical file, or streaming projection is invalid."""


def _canonical_relative(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CoverageBridgeError(f"{label} is not a canonical relative path")
    raw = PurePosixPath(value)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise CoverageBridgeError(f"{label} is not a canonical relative path")
    if raw.as_posix() != value:
        raise CoverageBridgeError(f"{label} is not a canonical relative path")
    return value


def _binding_path_sha(binding: object, *, label: str) -> tuple[str, str]:
    if not isinstance(binding, Mapping):
        raise CoverageBridgeError(f"{label} binding is absent")
    path = _canonical_relative(binding.get("path"), label=f"{label} path")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise CoverageBridgeError(f"{label} SHA-256 is malformed")
    return path, digest


def _state_relative(state: Mapping[str, Any], key: str) -> str:
    return _canonical_relative(state.get(key), label=f"state {key}")


def canonical_coverage_source_paths(
    authorization: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the eleven logical paths without consulting an audit document."""
    state = authorization.get("state_paths")
    registries = authorization.get("registries")
    if not isinstance(state, Mapping) or not isinstance(registries, Mapping):
        raise CoverageBridgeError("authorization lacks state/registry paths")
    policy_path, _ = _binding_path_sha(
        authorization.get("temporal_coverage_policy"),
        label="temporal coverage policy",
    )
    protocol_path, _ = _binding_path_sha(
        authorization.get("protocol"), label="protocol"
    )
    development_path, _ = _binding_path_sha(
        registries.get("development"), label="development registry"
    )
    external_path, _ = _binding_path_sha(
        registries.get("external"), label="external registry"
    )
    output = {
        "policy": policy_path,
        "protocol": protocol_path,
        "acquisition_manifest": _state_relative(state, "acquisition_manifest"),
        "temporal_normalized_outcomes": _state_relative(
            state, "temporal_outcomes"
        ),
        "external_normalized_outcomes": _state_relative(
            state, "external_outcomes"
        ),
        "temporal_site_registry": development_path,
        "external_site_registry": external_path,
        "temporal_full_predictions": _state_relative(
            state, "temporal_predictions"
        ),
        "external_full_predictions": _state_relative(
            state, "external_predictions"
        ),
        "availability_registry": _state_relative(
            state, "availability_registry"
        ),
        "statistics": _state_relative(state, "statistics"),
    }
    if tuple(output) != SOURCE_BINDING_KEYS:
        raise CoverageBridgeError("coverage source path registry order changed")
    if len(set(output.values())) != len(output):
        raise CoverageBridgeError("coverage logical source paths alias each other")
    return output


def _metadata_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class _PhysicalReader:
    """Open root-confined files through no-follow descriptors and detect aliasing."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._key_paths: dict[str, Path] = {}
        self._physical_owners: dict[Path, str] = {}
        self._inode_owners: dict[tuple[int, int], str] = {}
        self.digests: dict[str, str] = {}

    def _open_root(self) -> int:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(os.path.sep, flags)
        try:
            for component in self.root.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise CoverageBridgeError("coverage repository root is not a directory")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @contextmanager
    def open(
        self,
        *,
        key: str,
        logical_relative: str,
        physical_path: Path,
        expected_sha256: str | None,
    ) -> Iterator[BinaryIO]:
        logical_relative = _canonical_relative(
            logical_relative, label=f"coverage {key} logical path"
        )
        lexical = Path(os.path.abspath(os.fspath(physical_path)))
        try:
            relative = lexical.relative_to(self.root)
        except ValueError as exc:
            raise CoverageBridgeError(
                f"coverage {key} physical path escapes repository root"
            ) from exc
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise CoverageBridgeError(f"coverage {key} physical path is noncanonical")
        prior = self._key_paths.get(key)
        if prior is not None and prior != lexical:
            raise CoverageBridgeError(f"coverage {key} changed physical path between passes")
        owner = self._physical_owners.get(lexical)
        if owner is not None and owner != key:
            raise CoverageBridgeError(
                f"coverage physical path aliases {owner} and {key}"
            )
        self._key_paths[key] = lexical
        self._physical_owners[lexical] = key
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        root_descriptor = self._open_root()
        parent_descriptor = root_descriptor
        descriptor: int | None = None
        handle: BinaryIO | None = None
        before: os.stat_result | None = None
        try:
            for component in relative.parts[:-1]:
                child = os.open(component, directory_flags, dir_fd=parent_descriptor)
                if parent_descriptor != root_descriptor:
                    os.close(parent_descriptor)
                parent_descriptor = child
            parent_metadata = os.fstat(parent_descriptor)
            descriptor = os.open(
                relative.parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_descriptor,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_dev != parent_metadata.st_dev
            ):
                raise CoverageBridgeError(
                    f"coverage {key} is not one owner-held regular file"
                )
            inode = (before.st_dev, before.st_ino)
            inode_owner = self._inode_owners.get(inode)
            if inode_owner is not None and inode_owner != key:
                raise CoverageBridgeError(
                    f"coverage inode aliases {inode_owner} and {key}"
                )
            self._inode_owners[inode] = key
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if (
                expected_sha256 is not None
                and actual_sha256 != expected_sha256
            ):
                raise CoverageBridgeError(f"coverage {key} SHA-256 changed")
            earlier_digest = self.digests.get(key)
            if earlier_digest is not None and earlier_digest != actual_sha256:
                raise CoverageBridgeError(
                    f"coverage {key} bytes changed between streaming passes"
                )
            self.digests[key] = actual_sha256
            os.lseek(descriptor, 0, os.SEEK_SET)
            handle = os.fdopen(descriptor, "rb", closefd=False)
            yield handle
        except OSError as exc:
            raise CoverageBridgeError(
                f"coverage {key} physical path is absent or unsafe"
            ) from exc
        finally:
            if handle is not None:
                handle.close()
            metadata_changed = False
            if descriptor is not None:
                if before is not None:
                    after = os.fstat(descriptor)
                    metadata_changed = (
                        _metadata_signature(after) != _metadata_signature(before)
                    )
                os.close(descriptor)
            if parent_descriptor != root_descriptor:
                os.close(parent_descriptor)
            os.close(root_descriptor)
            if metadata_changed:
                raise CoverageBridgeError(
                    f"coverage {key} metadata changed during same-fd read"
                )


def _read_json(
    reader: _PhysicalReader,
    *,
    key: str,
    logical: str,
    physical: Path,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], bytes]:
    with reader.open(
        key=key,
        logical_relative=logical,
        physical_path=physical,
        expected_sha256=expected_sha256,
    ) as handle:
        payload = handle.read()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoverageBridgeError(f"coverage {key} JSON is malformed") from exc
    if not isinstance(value, dict):
        raise CoverageBridgeError(f"coverage {key} JSON is not an object")
    return value, payload


def _read_site_registry(
    reader: _PhysicalReader,
    *,
    key: str,
    logical: str,
    physical: Path,
    expected_sha256: str,
) -> tuple[str, ...]:
    with reader.open(
        key=key,
        logical_relative=logical,
        physical_path=physical,
        expected_sha256=expected_sha256,
    ) as handle:
        payload = handle.read()
    try:
        frame = pd.read_csv(
            io.BytesIO(payload), dtype={"site_no": "string"}, keep_default_na=False
        )
    except Exception as exc:
        raise CoverageBridgeError(f"coverage {key} CSV is malformed") from exc
    if "site_no" not in frame:
        raise CoverageBridgeError(f"coverage {key} lacks site_no")
    sites = tuple(sorted(frame.site_no.astype(str).str.strip()))
    if any(not site for site in sites) or len(sites) != len(set(sites)):
        raise CoverageBridgeError(f"coverage {key} site identity is invalid")
    return sites


def _read_observability(
    reader: _PhysicalReader,
    *,
    key: str,
    logical: str,
    physical: Path,
    expected_sha256: str,
    sites: Sequence[str],
) -> pd.DataFrame:
    columns = ["site_no", "DATE", "WTEMP", "WTEMP_value_status"]
    expected_types: Mapping[str, pa.DataType] = {
        "site_no": pa.string(),
        "DATE": pa.timestamp("ns"),
        "WTEMP": pa.float64(),
        "WTEMP_value_status": pa.string(),
    }
    try:
        with reader.open(
            key=key,
            logical_relative=logical,
            physical_path=physical,
            expected_sha256=expected_sha256,
        ) as handle:
            parquet = pq.ParquetFile(handle)
            schema = parquet.schema_arrow
            for column, expected_type in expected_types.items():
                if schema.names.count(column) != 1:
                    raise CoverageBridgeError(
                        f"coverage {key} outcome schema must contain one "
                        f"{column} field"
                    )
                actual_type = schema.field(column).type
                if actual_type != expected_type:
                    raise CoverageBridgeError(
                        f"coverage {key} outcome schema for {column} must be "
                        f"{expected_type}, not {actual_type}"
                    )
            frame = parquet.read(columns=columns).to_pandas()
    except CoverageBridgeError:
        raise
    except Exception as exc:
        raise CoverageBridgeError(f"coverage {key} parquet is malformed") from exc
    if list(frame.columns) != columns:
        raise CoverageBridgeError(f"coverage {key} projection schema changed")
    frame = frame.copy()
    if (
        frame.site_no.isna().any()
        or not frame.site_no.map(lambda value: isinstance(value, str)).all()
        or not frame.site_no.eq(frame.site_no.str.strip()).all()
        or frame.WTEMP_value_status.isna().any()
        or not frame.WTEMP_value_status.map(
            lambda value: isinstance(value, str)
        ).all()
        or not frame.WTEMP_value_status.eq(
            frame.WTEMP_value_status.str.strip()
        ).all()
        or frame.DATE.isna().any()
        or getattr(frame.DATE.dt, "tz", None) is not None
        or not frame.DATE.eq(frame.DATE.dt.normalize()).all()
        or frame.duplicated(["site_no", "DATE"]).any()
    ):
        raise CoverageBridgeError(f"coverage {key} has invalid site/date keys")
    values = frame.WTEMP.to_numpy(float)
    finite = np.isfinite(values)
    retained = frame.WTEMP_value_status.eq("RETAINED_FINITE_VALUE").to_numpy(
        bool
    )
    allowed_status = {
        "RETAINED_FINITE_VALUE",
        "MISSING_NO_FINITE_SERIES",
        "MULTIPLE_FINITE_SERIES_CONFLICT",
    }
    if (
        not set(frame.WTEMP_value_status) <= allowed_status
        or not np.array_equal(finite, retained)
    ):
        raise CoverageBridgeError(
            f"coverage {key} WTEMP status disagrees with finite values"
        )
    expected_dates = pd.date_range(CONSTRUCTION_BUFFER_START, TARGET_END, freq="D")
    expected = pd.MultiIndex.from_product(
        [tuple(sorted(sites)), expected_dates], names=["site_no", "DATE"]
    )
    actual = pd.MultiIndex.from_frame(frame[["site_no", "DATE"]])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise CoverageBridgeError(
            f"coverage {key} lacks the exact 2020-11-30 construction buffer/calendar"
        )
    projected = frame.loc[
        frame.DATE.ge(pd.Timestamp(CORE_PROJECTION_START)),
        ["site_no", "DATE"],
    ].copy()
    projected["wtemp_observed"] = retained[
        frame.DATE.ge(pd.Timestamp(CORE_PROJECTION_START)).to_numpy(bool)
    ]
    projected = projected.rename(columns={"site_no": "site_id", "DATE": "date"})
    if projected.date.min() != pd.Timestamp(CORE_PROJECTION_START):
        raise CoverageBridgeError("coverage core projection retained the construction row")
    return projected


def _read_availability(
    reader: _PhysicalReader,
    *,
    logical: str,
    physical: Path,
    expected_sha256: str | None,
) -> pd.DataFrame:
    """Read the schema-less CSV while rejecting coercible scalar aliases.

    CSV has no embedded string type, so site identifiers are deliberately read
    as strings to preserve leading zeroes.  In contrast, the producer writes
    horizon/count as canonical decimal integers and reportable as True/False;
    pandas' inferred physical dtypes must therefore be int64/int64/bool.
    """
    with reader.open(
        key="availability_registry",
        logical_relative=logical,
        physical_path=physical,
        expected_sha256=expected_sha256,
    ) as handle:
        payload = handle.read()
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            dtype={"site_no": "string"},
            keep_default_na=False,
        )
    except Exception as exc:
        raise CoverageBridgeError("coverage availability CSV is malformed") from exc
    if list(frame.columns) != list(AVAILABILITY_COLUMNS):
        raise CoverageBridgeError("coverage availability CSV schema changed")
    if (
        frame.cohort.isna().any()
        or not frame.cohort.map(lambda value: isinstance(value, str)).all()
        or not frame.cohort.eq(frame.cohort.str.strip()).all()
        or frame.site_no.isna().any()
        or not frame.site_no.map(lambda value: isinstance(value, str)).all()
        or not frame.site_no.eq(frame.site_no.str.strip()).all()
        or frame.horizon.dtype != np.dtype("int64")
        or frame.n_valid_targets.dtype != np.dtype("int64")
        or frame.reportable.dtype != np.dtype("bool")
    ):
        raise CoverageBridgeError(
            "coverage availability CSV violates the exact physical type contract"
        )
    return frame


def _new_row_hasher(domain: str) -> Any:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\n")
    return digest


def _update_row_hasher(digest: Any, row: Sequence[object]) -> None:
    digest.update(canonical_json(list(row)).encode("utf-8"))
    digest.update(b"\n")


def _normal_prediction_row(
    row: tuple[Any, ...],
) -> tuple[str, str, int, pd.Timestamp, pd.Timestamp, float, float]:
    raw_model, raw_site, raw_horizon, raw_issue, raw_target, raw_true, raw_pred = row
    if (
        not isinstance(raw_model, str)
        or not isinstance(raw_site, str)
        or not raw_model
        or not raw_site
        or raw_model != raw_model.strip()
        or raw_site != raw_site.strip()
        or isinstance(raw_horizon, (bool, np.bool_))
        or not isinstance(raw_horizon, (int, np.integer))
        or not isinstance(raw_issue, pd.Timestamp)
        or not isinstance(raw_target, pd.Timestamp)
        or isinstance(raw_true, (bool, np.bool_))
        or isinstance(raw_pred, (bool, np.bool_))
        or not isinstance(raw_true, (float, np.floating))
        or not isinstance(raw_pred, (float, np.floating))
    ):
        raise CoverageBridgeError("coverage prediction identity is malformed")
    model = raw_model
    site = raw_site
    horizon = int(raw_horizon)
    issue = raw_issue
    target = raw_target
    y_true = float(raw_true)
    y_pred = float(raw_pred)
    if (
        horizon not in HORIZONS
        or issue.tzinfo is not None
        or target.tzinfo is not None
        or issue != issue.normalize()
        or target != target.normalize()
        or target - issue != pd.Timedelta(days=horizon)
        or issue < pd.Timestamp(TARGET_START)
        or target > pd.Timestamp(TARGET_END)
        or not np.isfinite(y_true)
        or not np.isfinite(y_pred)
    ):
        raise CoverageBridgeError("coverage prediction leaves the frozen contract")
    return model, site, horizon, issue, target, y_true, y_pred


_PREDICTION_ARROW_TYPES: Mapping[str, pa.DataType] = {
    "model": pa.string(),
    "site_id": pa.string(),
    "horizon": pa.int64(),
    "issue_date": pa.timestamp("ns"),
    "target_date": pa.timestamp("ns"),
    "y_true": pa.float64(),
    "y_pred": pa.float64(),
}


def _validate_prediction_arrow_schema(
    parquet: pq.ParquetFile, *, key: str
) -> None:
    """Require the producer's exact physical types before any coercive read."""
    schema = parquet.schema_arrow
    for column, expected_type in _PREDICTION_ARROW_TYPES.items():
        if schema.names.count(column) != 1:
            raise CoverageBridgeError(
                f"coverage {key} prediction schema must contain one {column} field"
            )
        actual_type = schema.field(column).type
        if actual_type != expected_type:
            raise CoverageBridgeError(
                f"coverage {key} prediction schema for {column} must be "
                f"{expected_type}, not {actual_type}"
            )


def _stream_prediction_pass_one(
    reader: _PhysicalReader,
    *,
    key: str,
    logical: str,
    physical: Path,
    expected_sha256: str | None,
    cohort: str,
    sites: set[str],
) -> list[dict[str, Any]]:
    models = MODEL_REGISTRY[cohort]
    key_hashers = {model: _new_row_hasher(FORECAST_KEY_DIGEST_DOMAIN) for model in models}
    y_hashers = {model: _new_row_hasher(Y_TRUE_DIGEST_DOMAIN) for model in models}
    counts = {model: 0 for model in models}
    previous: tuple[str, str, int, pd.Timestamp, pd.Timestamp] | None = None
    columns = list(PREDICTION_COLUMNS)
    try:
        with reader.open(
            key=key,
            logical_relative=logical,
            physical_path=physical,
            expected_sha256=expected_sha256,
        ) as handle:
            parquet = pq.ParquetFile(handle)
            _validate_prediction_arrow_schema(parquet, key=key)
            for batch in parquet.iter_batches(
                batch_size=PARQUET_BATCH_ROWS, columns=columns
            ):
                frame = batch.to_pandas()
                for raw in frame.itertuples(index=False, name=None):
                    model, site, horizon, issue, target, y_true, _ = (
                        _normal_prediction_row(raw)
                    )
                    if model not in counts or site not in sites:
                        raise CoverageBridgeError(
                            f"coverage {key} contains an unknown model/site"
                        )
                    order = (model, site, horizon, issue, target)
                    if previous is not None and order <= previous:
                        raise CoverageBridgeError(
                            f"coverage {key} is not strictly stable-sorted"
                        )
                    previous = order
                    digest_key = (
                        site,
                        horizon,
                        issue.strftime("%Y-%m-%d"),
                        target.strftime("%Y-%m-%d"),
                    )
                    _update_row_hasher(key_hashers[model], digest_key)
                    _update_row_hasher(
                        y_hashers[model], (*digest_key, format(y_true, ".17g"))
                    )
                    counts[model] += 1
    except CoverageBridgeError:
        raise
    except Exception as exc:
        raise CoverageBridgeError(f"coverage {key} streaming pass one failed") from exc
    return [
        {
            "model": model,
            "row_count": counts[model],
            "forecast_key_sha256": key_hashers[model].hexdigest(),
            "y_true_sha256": y_hashers[model].hexdigest(),
        }
        for model in models
    ]


def _stream_temporal_comparisons(
    reader: _PhysicalReader,
    *,
    key: str,
    logical: str,
    physical: Path,
    expected_sha256: str | None,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    columns = list(PREDICTION_COLUMNS)
    try:
        with reader.open(
            key=key,
            logical_relative=logical,
            physical_path=physical,
            expected_sha256=expected_sha256,
        ) as handle:
            parquet = pq.ParquetFile(handle)
            _validate_prediction_arrow_schema(parquet, key=key)
            for batch in parquet.iter_batches(
                batch_size=PARQUET_BATCH_ROWS, columns=columns
            ):
                frame = batch.to_pandas()
                mask = frame.model.isin(COMPARISON_MODELS)
                if mask.any():
                    selected.append(frame.loc[mask, columns].copy())
    except CoverageBridgeError:
        raise
    except Exception as exc:
        raise CoverageBridgeError(
            "coverage temporal prediction streaming pass two failed"
        ) from exc
    if not selected:
        return pd.DataFrame(columns=columns)
    return pd.concat(selected, ignore_index=True)


def _expected_receipt_sha(
    receipt_artifacts: Mapping[str, Any] | None,
    *,
    source_key: str,
    logical_path: str,
) -> str | None:
    if receipt_artifacts is None:
        return None
    receipt_key = _RECEIPT_SOURCE_KEYS.get(source_key)
    if receipt_key is None:
        return None
    path, digest = _binding_path_sha(
        receipt_artifacts.get(receipt_key), label=f"receipt {receipt_key}"
    )
    if path != logical_path:
        raise CoverageBridgeError(f"receipt {receipt_key} path is noncanonical")
    return digest


def replay_temporal_coverage_from_physical_files(
    *,
    root: str | Path,
    authorization: Mapping[str, Any],
    trusted_physical_directory: str | Path | None = None,
    receipt_artifacts: Mapping[str, Any] | None = None,
    expected_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently rebuild all coverage inputs and replay the pure core."""
    root_path = Path(root).resolve()
    logical = canonical_coverage_source_paths(authorization)
    state = authorization.get("state_paths")
    registries = authorization.get("registries")
    if not isinstance(state, Mapping) or not isinstance(registries, Mapping):
        raise CoverageBridgeError("authorization lacks coverage dependencies")
    plan = authorization.get("acquisition_plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("history_start") != CONSTRUCTION_BUFFER_START
        or plan.get("target_start") != TARGET_START
        or plan.get("target_end") != TARGET_END
    ):
        raise CoverageBridgeError("coverage acquisition calendar changed")
    physical = {key: root_path / value for key, value in logical.items()}
    if trusted_physical_directory is not None:
        trusted_directory = Path(
            os.path.abspath(os.fspath(trusted_physical_directory))
        )
        for source_key in _TRUSTED_SOURCE_STATE_KEYS:
            physical[source_key] = trusted_directory / Path(
                logical[source_key]
            ).name
    reader = _PhysicalReader(root_path)

    policy_path, policy_expected = _binding_path_sha(
        authorization.get("temporal_coverage_policy"),
        label="temporal coverage policy",
    )
    if policy_path != POLICY_RELATIVE or policy_expected != POLICY_FILE_SHA256:
        raise CoverageBridgeError("authorized temporal coverage policy changed")
    _, policy_payload = _read_json(
        reader,
        key="policy",
        logical=logical["policy"],
        physical=physical["policy"],
        expected_sha256=policy_expected,
    )
    policy = validate_temporal_coverage_policy_bytes(policy_payload)

    protocol_path, protocol_expected = _binding_path_sha(
        authorization.get("protocol"), label="protocol"
    )
    if protocol_path != logical["protocol"]:
        raise CoverageBridgeError("authorized protocol path changed")
    protocol, _ = _read_json(
        reader,
        key="protocol",
        logical=logical["protocol"],
        physical=physical["protocol"],
        expected_sha256=protocol_expected,
    )

    acquisition_expected = _expected_receipt_sha(
        receipt_artifacts,
        source_key="acquisition_manifest",
        logical_path=logical["acquisition_manifest"],
    )
    acquisition, _ = _read_json(
        reader,
        key="acquisition_manifest",
        logical=logical["acquisition_manifest"],
        physical=physical["acquisition_manifest"],
        expected_sha256=acquisition_expected,
    )
    normalized_bindings = acquisition.get("normalized_outcome_tables")
    if not isinstance(normalized_bindings, Mapping) or set(normalized_bindings) != {
        "temporal",
        "external",
    }:
        raise CoverageBridgeError("acquisition manifest lacks normalized outcomes")

    development_path, development_sha = _binding_path_sha(
        registries.get("development"), label="development registry"
    )
    external_path, external_sha = _binding_path_sha(
        registries.get("external"), label="external registry"
    )
    if (
        development_path != logical["temporal_site_registry"]
        or external_path != logical["external_site_registry"]
    ):
        raise CoverageBridgeError("coverage site registry path changed")
    sites = {
        "temporal": _read_site_registry(
            reader,
            key="temporal_site_registry",
            logical=logical["temporal_site_registry"],
            physical=physical["temporal_site_registry"],
            expected_sha256=development_sha,
        ),
        "external": _read_site_registry(
            reader,
            key="external_site_registry",
            logical=logical["external_site_registry"],
            physical=physical["external_site_registry"],
            expected_sha256=external_sha,
        ),
    }
    if set(sites["temporal"]) & set(sites["external"]):
        raise CoverageBridgeError("coverage cohorts overlap")

    observability: dict[str, pd.DataFrame] = {}
    for cohort in ("temporal", "external"):
        source_key = f"{cohort}_normalized_outcomes"
        manifest_path, manifest_sha = _binding_path_sha(
            normalized_bindings.get(cohort),
            label=f"acquisition normalized {cohort}",
        )
        if manifest_path != logical[source_key]:
            raise CoverageBridgeError(
                f"acquisition normalized {cohort} path is noncanonical"
            )
        receipt_sha = _expected_receipt_sha(
            receipt_artifacts,
            source_key=source_key,
            logical_path=logical[source_key],
        )
        if receipt_sha is not None and receipt_sha != manifest_sha:
            raise CoverageBridgeError(
                f"receipt/acquisition normalized {cohort} SHA-256 differs"
            )
        observability[cohort] = _read_observability(
            reader,
            key=source_key,
            logical=logical[source_key],
            physical=physical[source_key],
            expected_sha256=manifest_sha,
            sites=sites[cohort],
        )

    model_registry = authorization.get("required_models")
    if not isinstance(model_registry, Mapping) or {
        cohort: tuple(model_registry.get(cohort, ()))
        for cohort in ("temporal", "external")
    } != MODEL_REGISTRY:
        raise CoverageBridgeError("authorized coverage model registry changed")

    model_key_audits: dict[str, list[dict[str, Any]]] = {}
    for cohort in ("temporal", "external"):
        source_key = f"{cohort}_full_predictions"
        expected_sha = _expected_receipt_sha(
            receipt_artifacts,
            source_key=source_key,
            logical_path=logical[source_key],
        )
        model_key_audits[cohort] = _stream_prediction_pass_one(
            reader,
            key=source_key,
            logical=logical[source_key],
            physical=physical[source_key],
            expected_sha256=expected_sha,
            cohort=cohort,
            sites=set(sites[cohort]),
        )
    temporal_comparisons = _stream_temporal_comparisons(
        reader,
        key="temporal_full_predictions",
        logical=logical["temporal_full_predictions"],
        physical=physical["temporal_full_predictions"],
        expected_sha256=_expected_receipt_sha(
            receipt_artifacts,
            source_key="temporal_full_predictions",
            logical_path=logical["temporal_full_predictions"],
        ),
    )

    availability_expected = _expected_receipt_sha(
        receipt_artifacts,
        source_key="availability_registry",
        logical_path=logical["availability_registry"],
    )
    availability = _read_availability(
        reader,
        logical=logical["availability_registry"],
        physical=physical["availability_registry"],
        expected_sha256=availability_expected,
    )

    statistics_expected = _expected_receipt_sha(
        receipt_artifacts,
        source_key="statistics",
        logical_path=logical["statistics"],
    )
    statistics, _ = _read_json(
        reader,
        key="statistics",
        logical=logical["statistics"],
        physical=physical["statistics"],
        expected_sha256=statistics_expected,
    )
    tests = statistics.get("tests")
    family = protocol.get("primary_inference_contract", {}).get(
        "confirmatory_family"
    )
    if not isinstance(tests, list) or not isinstance(family, list):
        raise CoverageBridgeError("coverage formal test/statistics projection is absent")

    source_bindings = {
        key: {"path": logical[key], "sha256": reader.digests[key]}
        for key in SOURCE_BINDING_KEYS
    }
    audit = build_temporal_coverage_audit(
        policy=policy,
        source_bindings=source_bindings,
        target_start=TARGET_START,
        target_end=TARGET_END,
        sites_by_cohort=sites,
        model_registry_by_cohort=MODEL_REGISTRY,
        model_key_audits_by_cohort=model_key_audits,
        observability_by_cohort=observability,
        temporal_comparison_predictions=temporal_comparisons,
        availability=availability,
        formal_tests=family,
        primary_statistics=tests,
    )
    if expected_audit is not None:
        try:
            validate_temporal_coverage_audit(
                expected_audit,
                policy=policy,
                source_bindings=source_bindings,
                target_start=TARGET_START,
                target_end=TARGET_END,
                sites_by_cohort=sites,
                model_registry_by_cohort=MODEL_REGISTRY,
                model_key_audits_by_cohort=model_key_audits,
                observability_by_cohort=observability,
                temporal_comparison_predictions=temporal_comparisons,
                availability=availability,
                formal_tests=family,
                primary_statistics=tests,
            )
        except Exception as exc:
            raise CoverageBridgeError(
                "temporal coverage audit differs from physical replay"
            ) from exc
    if audit.get("format") != AUDIT_FORMAT or audit.get("status") != (
        "DERIVED_CORE_REQUIRES_RECEIPT_BINDING"
    ):
        raise CoverageBridgeError("temporal coverage core status changed")
    return audit
