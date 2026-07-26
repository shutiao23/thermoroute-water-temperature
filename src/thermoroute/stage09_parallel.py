"""Fail-closed member-level precomputation for the Stage-09 controls.

The seven Stage-09 architecture controls are scientifically independent once
their shared, frozen inputs have been resolved.  This module separates that
scientific contract from the execution-only worker count and provides three
small transactions:

* freeze one exact 7-arm x 5-seed work-order registry after the live prelabel
  model-matrix gate has passed;
* publish each completed member as one create-only tar file containing a
  canonical receipt and every result byte; and
* freeze a coordinator receipt only after the exact 35-member matrix has been
  independently replayed (no missing, duplicate, or extra member is allowed).

The tar file is the member commit point.  Training happens in an unobservable
staging directory and publication uses the host kernel's same-filesystem,
atomic, no-replace rename primitive.  A crash therefore leaves either a private
staging object or one complete single-link archive, never a public two-link
intermediate.  A retry may reuse an existing member only after its complete
archive, receipt, and scientific semantics have been revalidated.  Malformed,
partial, colliding, or tampered final paths are never overwritten.

No outcome file, prediction value, metric, hostname, timestamp, PID, worker
count, or scheduling order enters the work order's scientific identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any

from .input_closure import resolve_development_input_closure
from .model_matrix_amendment import (
    AMENDMENT_FORMAT,
    AMENDMENT_ID,
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_FORMAT,
    AMENDMENT_SEAL_RELATIVE,
    AMENDMENT_SEAL_STATUS,
    AMENDMENT_STATUS,
    SEEDS as MODEL_MATRIX_SEEDS,
    STAGE09_CONTROLS,
    model_matrix_contract_id,
    validate_model_matrix_amendment,
    validate_model_matrix_amendment_seal,
)
from .repro import (
    RUN_SCHEMA_VERSION,
    RunIdentity,
    advisory_file_lock,
    assert_formal_numerical_policy,
    numerical_runtime_contract,
    resolve_run_identity,
    sha256_json,
    source_tree_hash,
)


AUTHORIZATION_FORMAT = "thermoroute.stage09-control-authorization.v1"
WORK_ORDER_FORMAT = "thermoroute.stage09-control-work-order.v1"
MEMBER_RECEIPT_FORMAT = "thermoroute.stage09-control-member-receipt.v1"
COORDINATOR_RECEIPT_FORMAT = (
    "thermoroute.stage09-control-coordinator-receipt.v1"
)
CONTROL_PRECOMPUTE_DIRECTORY = "stage09_control_precompute_v1"
CONTROL_RESUME_DIRECTORY = ".stage09-control-member-resume"
CONTROL_MEMBER_COUNT = 35
DEFAULT_CONTROL_WORKERS = 2
RECOMMENDED_MAX_CONTROL_WORKERS = 2
MAX_CONTROL_WORKERS = 8
MEMBER_ARCHIVE_SUFFIX = ".member.tar"
EXPECTED_MEMBER_PAYLOADS = (
    "bundle/metadata.json",
    "bundle/weights.pt",
    "predictions.parquet",
    "predictions.parquet.meta.json",
    "training_checkpoint.pt",
    "training_checkpoint.pt.meta.json",
)
FORMAL_RUN_CONFIG_FIELDS = frozenset(
    {
        "stage",
        "protocol",
        "panel",
        "station_registry",
        "variables",
        "horizons",
        "context_length",
        "seeds",
        "time_split",
        "train_config",
        "thermoroute_seeds",
        "lightgbm_seeds",
        "ablation_seeds",
        "delta_scale",
        "station_sampling",
        "selection_metric",
        "ablations",
        "air2stream",
        "device",
        "training_device",
        "execution_role",
        "development_predictor_bridge",
        "eval_batch_size",
        "lightgbm_validation_grid",
        "event_reference_fit_interval",
        "formal_numerical_policy",
        "input_closure_sha256",
        "input_closure_file_count",
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{20}")
_MAX_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 256
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024 * 1024
SEMANTIC_VALIDATION_FORMAT = (
    "thermoroute.stage09-control-semantic-validation.v2"
)


class Stage09ParallelError(RuntimeError):
    """A Stage-09 member plan, archive, or matrix failed closed."""


@dataclass(frozen=True, order=True)
class ControlMember:
    """One member of the frozen arm-major, seed-minor matrix."""

    arm_index: int
    arm_id: str
    seed: int

    @property
    def member_id(self) -> str:
        return f"arm{self.arm_index:02d}-seed{self.seed}"

    @property
    def archive_relative(self) -> str:
        return (
            f"members/arm{self.arm_index:02d}/"
            f"seed{self.seed}{MEMBER_ARCHIVE_SUFFIX}"
        )

    @property
    def work_order_relative(self) -> str:
        return f"work_orders/arm{self.arm_index:02d}/seed{self.seed}.json"

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "arm_index": self.arm_index,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "same_seed_reference": {
                "model_id": "ThermoRoute",
                "seed": self.seed,
            },
        }


@dataclass(frozen=True)
class ModelMatrixGate:
    """Outcome-free matrix objects and exact bytes validated by the parent."""

    amendment: Mapping[str, Any]
    seal: Mapping[str, Any]
    binding: Mapping[str, Any]
    contract_id: str


@dataclass(frozen=True)
class ValidatedWorkOrder:
    """A live-gated work order safe to hand to one scientific trainer."""

    root: Path
    path: Path
    document: Mapping[str, Any]
    authorization: Mapping[str, Any]
    identity: RunIdentity
    input_closure: Any
    matrix_gate: ModelMatrixGate

    @property
    def member(self) -> ControlMember:
        value = self.document["member"]
        assert isinstance(value, Mapping)
        return ControlMember(
            arm_index=int(value["arm_index"]),
            arm_id=str(value["arm_id"]),
            seed=int(value["seed"]),
        )

    @property
    def run_directory(self) -> Path:
        return _run_directory(self.root, self.identity.run_id)

    @property
    def resume_checkpoint_path(self) -> Path:
        """Return this work order's fixed, content-addressed resume path."""
        work_order_sha256 = self.document.get("work_order_self_sha256")
        if (
            type(work_order_sha256) is not str
            or _DIGEST.fullmatch(work_order_sha256) is None
        ):
            raise Stage09ParallelError("work order lacks its resume content address")
        return (
            self.run_directory
            / CONTROL_RESUME_DIRECTORY
            / self.member.member_id
            / work_order_sha256
            / "training_checkpoint.pt"
        )

    def assert_unchanged(self) -> None:
        """Recheck every live scientific identity input before publication."""
        assert_formal_numerical_policy()
        self.input_closure.assert_unchanged()
        if source_tree_hash(self.root) != self.identity.source_sha256:
            raise Stage09ParallelError(
                "Stage-09 source tree changed after member authorization"
            )
        if sha256_json(numerical_runtime_contract()) != self.identity.runtime_sha256:
            raise Stage09ParallelError(
                "Stage-09 numerical runtime changed after member authorization"
            )
        _assert_binding_bytes(
            self.root,
            self.matrix_gate.binding["amendment"],
            label="model-matrix amendment",
        )
        _assert_binding_bytes(
            self.root,
            self.matrix_gate.binding["seal"],
            label="model-matrix amendment seal",
        )


@dataclass(frozen=True)
class _PublicationGuardClosure:
    """Adapt the parent full-input guard to the work-order closure protocol."""

    guard: Callable[[], object]

    def assert_unchanged(self) -> None:
        self.guard()


def expected_control_members() -> tuple[ControlMember, ...]:
    """Return the exact frozen 7 x 5 registry in aggregation order."""
    members = tuple(
        ControlMember(arm_index, arm_id, int(seed))
        for arm_index, arm_id in enumerate(STAGE09_CONTROLS)
        for seed in MODEL_MATRIX_SEEDS
    )
    if len(members) != CONTROL_MEMBER_COUNT:  # pragma: no cover - constants
        raise AssertionError("Stage-09 control registry is not 7 x 5")
    return members


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage09ParallelError("value is not canonical JSON") from exc
    return encoded


def _with_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise Stage09ParallelError(f"self-hash field already exists: {field}")
    stable = dict(value)
    return {**stable, field: hashlib.sha256(_canonical_bytes(stable)).hexdigest()}


def _validate_self_hash(
    value: object,
    *,
    field: str,
    expected_keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys | {field}:
        raise Stage09ParallelError(f"{label} schema is not exact")
    digest = value.get(field)
    stable = {key: item for key, item in value.items() if key != field}
    if (
        type(digest) is not str
        or _DIGEST.fullmatch(digest) is None
        or digest != hashlib.sha256(_canonical_bytes(stable)).hexdigest()
    ):
        raise Stage09ParallelError(f"{label} self-hash differs")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_relative_file(root, relative)
    return {
        "path": relative,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _assert_binding_bytes(
    root: Path, binding: object, *, label: str
) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise Stage09ParallelError(f"{label} binding schema changed")
    relative = binding.get("path")
    digest = binding.get("sha256")
    byte_count = binding.get("bytes")
    if (
        type(relative) is not str
        or type(digest) is not str
        or _DIGEST.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise Stage09ParallelError(f"{label} binding is malformed")
    path = _safe_relative_file(root, relative)
    if path.stat().st_size != byte_count or _sha256_file(path) != digest:
        raise Stage09ParallelError(f"{label} bytes changed")
    return path


def _normal_relative(value: str, *, label: str) -> str:
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or "\\" in value
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise Stage09ParallelError(f"{label} is not a canonical relative path")
    return value


def _safe_relative_file(root: Path, relative: str) -> Path:
    relative = _normal_relative(relative, label="file path")
    root = root.resolve()
    current = root
    for index, part in enumerate(Path(relative).parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            raise Stage09ParallelError(f"required file is absent: {relative}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise Stage09ParallelError(f"path contains a symlink: {relative}")
        if index < len(Path(relative).parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise Stage09ParallelError(
                    f"path crosses a non-directory: {relative}"
                )
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise Stage09ParallelError(
                f"path is not a single-link regular file: {relative}"
            )
    return root / relative


def _relative_to_root(root: Path, path: Path, *, label: str) -> str:
    root = root.resolve()
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise Stage09ParallelError(f"{label} escapes the repository") from exc
    return _normal_relative(relative, label=label)


def _safe_directory(path: Path, *, parent: Path | None = None) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if parent is not None and lexical.parent != parent.resolve():
        raise Stage09ParallelError("directory has an unexpected parent")
    try:
        lexical.mkdir(mode=0o700)
    except FileExistsError:
        # Another member process may win the first-create race.  EEXIST is not
        # authority: inspect the object again below before accepting it.
        pass
    except OSError as exc:
        raise Stage09ParallelError(
            f"directory cannot be created safely: {lexical}"
        ) from exc
    try:
        info = lexical.lstat()
    except OSError as exc:
        raise Stage09ParallelError(
            f"directory cannot be revalidated: {lexical}"
        ) from exc
    owner_matches = not hasattr(os, "getuid") or info.st_uid == os.getuid()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or not owner_matches
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise Stage09ParallelError(f"unsafe directory collision: {lexical}")
    return lexical


def _mkdir_chain(root: Path, relative: str) -> Path:
    root = root.resolve()
    current = root
    for part in Path(_normal_relative(relative, label="directory path")).parts:
        current = _safe_directory(current / part, parent=current)
    return current


def _write_create_only(path: Path, content: bytes) -> None:
    """Atomically create exact bytes or accept an identical committed file.

    Writing directly to the final name exposes an empty/partial file to a
    concurrent controller that loses ``O_EXCL``.  Instead, fully fsync one
    owner-private same-directory candidate and publish it with the same kernel
    no-replace primitive used for member archives.  The losing controller can
    therefore observe only an already complete file.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    parent = _safe_directory(path.parent)

    def accept_existing() -> None:
        try:
            info = path.lstat()
        except OSError as exc:
            raise Stage09ParallelError(
                f"create-only file cannot be inspected: {path}"
            ) from exc
        owner_matches = not hasattr(os, "getuid") or info.st_uid == os.getuid()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or not owner_matches
            or stat.S_IMODE(info.st_mode) & 0o077
            or path.read_bytes() != content
        ):
            raise Stage09ParallelError(f"create-only file collision: {path}")

    if os.path.lexists(path):
        accept_existing()
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".create-only.tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        try:
            _atomic_exclusive_rename(temporary, path)
        except Stage09ParallelError:
            if not os.path.lexists(path):
                raise
            accept_existing()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_exclusive_rename(source: Path, destination: Path) -> None:
    """Atomically move one private file to an absent public name.

    Python's ``os.rename`` may overwrite an existing destination and
    link-then-unlink exposes a crash window with two names for one inode.  The
    two supported formal hosts both provide a kernel no-replace rename:
    ``renamex_np(RENAME_EXCL)`` on Darwin and
    ``renameat2(RENAME_NOREPLACE)`` on Linux.  Unsupported hosts fail closed.
    """
    source = Path(os.path.abspath(os.fspath(source)))
    destination = Path(os.path.abspath(os.fspath(destination)))
    try:
        source_info = source.lstat()
        source_parent = source.parent.lstat()
        destination_parent = destination.parent.lstat()
    except OSError as exc:
        raise Stage09ParallelError(
            "member archive rename endpoints cannot be inspected"
        ) from exc
    if (
        not stat.S_ISREG(source_info.st_mode)
        or stat.S_ISLNK(source_info.st_mode)
        or source_info.st_nlink != 1
        or not stat.S_ISDIR(source_parent.st_mode)
        or stat.S_ISLNK(source_parent.st_mode)
        or not stat.S_ISDIR(destination_parent.st_mode)
        or stat.S_ISLNK(destination_parent.st_mode)
        or source_info.st_dev != destination_parent.st_dev
    ):
        raise Stage09ParallelError(
            "member archive rename endpoints are unsafe or cross-filesystem"
        )
    if os.path.lexists(destination):
        raise Stage09ParallelError(
            f"create-only member archive collision: {destination.name}"
        )

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:  # pragma: no cover - all supported macOS hosts
            raise Stage09ParallelError(
                "host lacks atomic exclusive renamex_np"
            )
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        return_code = renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - supported glibc exposes it
            raise Stage09ParallelError(
                "host lacks atomic exclusive renameat2"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        return_code = renameat2(
            -100, source_bytes, -100, destination_bytes, 0x00000001
        )
    else:  # pragma: no cover - formal numerical policy supports Darwin/Linux
        raise Stage09ParallelError(
            f"host {sys.platform!r} lacks a supported atomic exclusive rename"
        )
    if return_code != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise Stage09ParallelError(
                f"create-only member archive collision: {destination.name}"
            )
        if error_number == errno.EXDEV:
            raise Stage09ParallelError(
                "member staging and archive are not on one filesystem"
            )
        raise Stage09ParallelError(
            "atomic exclusive member archive rename failed"
        ) from OSError(error_number, os.strerror(error_number))

    try:
        committed = destination.lstat()
    except OSError as exc:
        raise Stage09ParallelError(
            "atomic member archive rename lacks its destination"
        ) from exc
    if (
        os.path.lexists(source)
        or not stat.S_ISREG(committed.st_mode)
        or stat.S_ISLNK(committed.st_mode)
        or committed.st_nlink != 1
        or committed.st_dev != source_info.st_dev
        or committed.st_ino != source_info.st_ino
    ):
        raise Stage09ParallelError(
            "atomic exclusive member archive rename postcondition failed"
        )
    _fsync_directory(destination.parent)
    if source.parent != destination.parent:
        _fsync_directory(source.parent)


def _load_canonical_json_bytes(
    raw: bytes,
    *,
    label: str,
    repository_style: bool = False,
) -> Any:
    """Load one canonical JSON line, rejecting duplicates and nonfinite values."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage09ParallelError(
                    f"{label} contains a duplicate JSON key: {key}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise Stage09ParallelError(
            f"{label} contains a nonfinite JSON value: {value}"
        )

    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage09ParallelError(f"{label} is not valid JSON") from exc
    expected = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        if repository_style
        else _canonical_bytes(value) + b"\n"
    )
    if raw != expected:
        raise Stage09ParallelError(
            f"{label} bytes are not the exact canonical JSON line"
        )
    return value


def _load_json(
    path: Path,
    *,
    label: str,
    maximum: int = _MAX_RECEIPT_BYTES,
    repository_style: bool = False,
) -> Any:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_size > maximum
        ):
            raise Stage09ParallelError(f"{label} is not a bounded regular file")
        return _load_canonical_json_bytes(
            path.read_bytes(),
            label=label,
            repository_style=repository_style,
        )
    except OSError as exc:
        raise Stage09ParallelError(f"{label} cannot be read") from exc


def _run_directory(root: Path, run_id: str) -> Path:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise Stage09ParallelError("Stage-09 run_id is not canonical")
    return (
        root.resolve()
        / "outputs"
        / "runs"
        / "09_usgs_experiment"
        / run_id
    )


def _control_root(root: Path, run_id: str) -> Path:
    return _run_directory(root, run_id) / CONTROL_PRECOMPUTE_DIRECTORY


def _resume_sidecar(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.name + ".meta.json")


def _resume_transaction_name_allowed(name: str) -> bool:
    canonical = ("training_checkpoint.pt", "training_checkpoint.pt.meta.json")
    return name in canonical or (
        name.endswith(".tmp")
        and any(name.startswith(f".{base}.") for base in canonical)
    )


def _prepare_member_resume_checkpoint(validated: ValidatedWorkOrder) -> Path:
    """Create and revalidate one work-order-addressed resume namespace.

    The member advisory lock must already be held.  Atomic checkpoint writers
    may leave only their exact owner-private ``.tmp`` names after an abrupt
    process death; the checkpoint loader decides whether those files represent
    a recoverable transaction.
    """
    checkpoint = validated.resume_checkpoint_path
    resume_root = _safe_directory(
        validated.run_directory / CONTROL_RESUME_DIRECTORY,
        parent=validated.run_directory,
    )
    member_root = _safe_directory(
        resume_root / validated.member.member_id,
        parent=resume_root,
    )
    address_root = _safe_directory(
        member_root / str(validated.document["work_order_self_sha256"]),
        parent=member_root,
    )
    if checkpoint.parent != address_root:
        raise Stage09ParallelError("resume checkpoint escaped its content address")
    for entry in address_root.iterdir():
        try:
            info = entry.lstat()
        except OSError as exc:
            raise Stage09ParallelError(
                "resume checkpoint namespace cannot be inspected"
            ) from exc
        owner_matches = not hasattr(os, "getuid") or info.st_uid == os.getuid()
        if (
            not _resume_transaction_name_allowed(entry.name)
            or not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_nlink != 1
            or not owner_matches
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise Stage09ParallelError(
                f"unsafe member resume checkpoint entry: {entry.name}"
            )
    return checkpoint


def materialize_member_resume_checkpoint(
    *, validated: ValidatedWorkOrder, payload: Path
) -> tuple[Path, Path]:
    """Copy one completed persistent checkpoint pair into a private payload."""
    checkpoint = _prepare_member_resume_checkpoint(validated)
    sidecar = _resume_sidecar(checkpoint)
    names = {entry.name for entry in checkpoint.parent.iterdir()}
    if names != {checkpoint.name, sidecar.name}:
        raise Stage09ParallelError(
            "completed member checkpoint transaction is not an exact pair"
        )
    destinations = (
        payload / "training_checkpoint.pt",
        payload / "training_checkpoint.pt.meta.json",
    )
    for source, destination in zip(
        (checkpoint, sidecar), destinations, strict=True
    ):
        before = source.lstat()
        content = source.read_bytes()
        after = source.lstat()
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise Stage09ParallelError(
                "resume checkpoint changed while materializing its archive"
            )
        _write_create_only(destination, content)
    return destinations


def _cleanup_member_resume_checkpoint(validated: ValidatedWorkOrder) -> None:
    """Remove only validated member-private resume state after archive commit."""
    checkpoint = _prepare_member_resume_checkpoint(validated)
    address_root = checkpoint.parent
    for entry in sorted(address_root.iterdir(), key=lambda item: item.name):
        # _prepare_member_resume_checkpoint has already checked owner, type,
        # link count, permissions, and the exact transaction-name grammar.
        entry.unlink()
    _fsync_directory(address_root)
    member_root = address_root.parent
    resume_root = member_root.parent
    address_root.rmdir()
    _fsync_directory(member_root)
    if not any(member_root.iterdir()):
        member_root.rmdir()
        _fsync_directory(resume_root)


def _validate_matrix_document(document: Mapping[str, Any]) -> Mapping[str, Any]:
    matrix = document.get("stage09_architecture_control_matrix")
    if not isinstance(matrix, Mapping):
        raise Stage09ParallelError("model-matrix amendment lacks Stage-09 matrix")
    required = {
        "stage",
        "role",
        "control_arms",
        "arm_count",
        "seeds_per_arm",
        "seed_count_per_arm",
        "expected_member_count",
        "matrix_identity",
        "pairing",
        "reporting",
    }
    pairing = matrix.get("pairing")
    reporting = matrix.get("reporting")
    if (
        set(matrix) != required
        or matrix.get("stage") != "09"
        or matrix.get("role") != "MANDATORY_EXPLORATORY_ARCHITECTURE_CONTROLS"
        or tuple(matrix.get("control_arms", ())) != tuple(STAGE09_CONTROLS)
        or matrix.get("arm_count") != len(STAGE09_CONTROLS)
        or tuple(matrix.get("seeds_per_arm", ())) != tuple(MODEL_MATRIX_SEEDS)
        or matrix.get("seed_count_per_arm") != len(MODEL_MATRIX_SEEDS)
        or matrix.get("expected_member_count") != CONTROL_MEMBER_COUNT
        or matrix.get("matrix_identity")
        != "7_control_arms_x_5_fixed_seeds_equals_35_members"
        or not isinstance(pairing, Mapping)
        or pairing.get("reference")
        != "ThermoRoute_member_with_the_exact_same_seed"
        or pairing.get("same_seed_required") is not True
        or pairing.get("exact_forecast_keys_required") is not True
        or pairing.get("exact_y_true_required") is not True
        or pairing.get("best_seed_selection_allowed") is not False
        or not isinstance(reporting, Mapping)
        or reporting.get("each_seed_retained") is not True
        or reporting.get("equal_weight_five_member_ensemble_mean_required")
        is not True
    ):
        raise Stage09ParallelError("Stage-09 model-matrix contract changed")
    return matrix


def validate_stage09_model_matrix_gate(root: str | Path) -> ModelMatrixGate:
    """Replay the exact outcome-free amendment, seal, bytes, and Git lineage."""
    repository = Path(root).resolve()
    try:
        amendment = validate_model_matrix_amendment(
            repository / AMENDMENT_RELATIVE,
            root=repository,
        )
        seal = validate_model_matrix_amendment_seal(
            repository / AMENDMENT_SEAL_RELATIVE,
            root=repository,
            amendment_path=repository / AMENDMENT_RELATIVE,
        )
    except Exception as exc:
        raise Stage09ParallelError("Stage-09 model-matrix gate failed") from exc
    matrix = _validate_matrix_document(amendment)
    stage09b = amendment.get("stage09b_development_control_matrix")
    if not isinstance(stage09b, Mapping):
        raise Stage09ParallelError("model-matrix amendment lacks Stage-09b matrix")
    contract_id = model_matrix_contract_id(matrix, stage09b)
    binding = {
        "format": "thermoroute.stage09-control-model-matrix-binding.v1",
        "amendment": _file_binding(repository, AMENDMENT_RELATIVE),
        "seal": _file_binding(repository, AMENDMENT_SEAL_RELATIVE),
        "amendment_format": AMENDMENT_FORMAT,
        "amendment_status": AMENDMENT_STATUS,
        "seal_format": AMENDMENT_SEAL_FORMAT,
        "seal_status": AMENDMENT_SEAL_STATUS,
        "amendment_id": AMENDMENT_ID,
        "amendment_document_commit": seal["amendment_document_commit"],
        "model_matrix_contract_id": contract_id,
    }
    return ModelMatrixGate(
        amendment=amendment,
        seal=seal,
        binding=binding,
        contract_id=contract_id,
    )


def _identity_from_mapping(value: object) -> RunIdentity:
    fields = {
        "run_id",
        "panel_sha256",
        "registry_sha256",
        "config_sha256",
        "source_sha256",
        "runtime_sha256",
        "input_closure_sha256",
        "schema_version",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Stage09ParallelError("run identity schema changed")
    try:
        identity = RunIdentity(**{key: str(item) for key, item in value.items()})
    except (TypeError, ValueError) as exc:
        raise Stage09ParallelError("run identity is malformed") from exc
    if (
        _RUN_ID.fullmatch(identity.run_id) is None
        or identity.schema_version != RUN_SCHEMA_VERSION
        or any(
            _DIGEST.fullmatch(getattr(identity, field)) is None
            for field in (
                "panel_sha256",
                "registry_sha256",
                "config_sha256",
                "source_sha256",
                "runtime_sha256",
                "input_closure_sha256",
            )
        )
    ):
        raise Stage09ParallelError("run identity digests are malformed")
    identity_parts = {
        key: item
        for key, item in identity.as_dict().items()
        if key != "run_id"
    }
    if sha256_json(identity_parts)[:20] != identity.run_id:
        raise Stage09ParallelError(
            "run identity is not its declared content address"
        )
    return identity


def _validate_formal_run_config(config: object) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise Stage09ParallelError("Stage-09 resolved config is not an object")
    if set(config) != FORMAL_RUN_CONFIG_FIELDS:
        raise Stage09ParallelError(
            "Stage-09 formal resolved-config schema changed"
        )
    required_values = {
        "stage": "09_usgs_experiment",
        "panel": "panel_usgs_120v2.parquet",
        "station_registry": "station_registry_v1.csv",
        "seeds": 5,
        "thermoroute_seeds": [0, 1, 2, 3, 4],
        "lightgbm_seeds": [0, 1, 2, 3, 4],
        "ablation_seeds": [0, 1, 2, 3, 4],
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "ablations": True,
        "air2stream": False,
        "device": "cpu",
        "training_device": "cpu",
        "execution_role": "route_a_formal_candidate",
    }
    for field, expected in required_values.items():
        actual = config.get(field)
        if isinstance(expected, list):
            actual = list(actual) if isinstance(actual, (list, tuple)) else actual
        if actual != expected:
            raise Stage09ParallelError(
                f"Stage-09 formal resolved config changed: {field}"
            )
    if (
        type(config.get("input_closure_sha256")) is not str
        or _DIGEST.fullmatch(str(config.get("input_closure_sha256"))) is None
        or type(config.get("input_closure_file_count")) is not int
        or config["input_closure_file_count"] < 1
        or config.get("formal_numerical_policy") is None
        or not isinstance(config.get("train_config"), Mapping)
        or not isinstance(config.get("development_predictor_bridge"), Mapping)
        or tuple(config.get("variables", ()))
        != ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
        or tuple(config.get("horizons", ())) != (1, 3, 7)
        or config.get("context_length") != 32
        or tuple(config.get("event_reference_fit_interval", ()))
        != ("2006-01-01", "2018-12-31")
        or not isinstance(config.get("delta_scale"), (int, float))
        or isinstance(config.get("delta_scale"), bool)
        or config.get("protocol")
        != f"route_a_strict_v1_balanced_delta{float(config['delta_scale']):g}"
    ):
        raise Stage09ParallelError("Stage-09 formal resolved config is incomplete")
    return config


def _scientific_member_config(
    config: Mapping[str, Any],
    member: ControlMember,
    intervention: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "algorithm": "thermoroute.train.fit_model",
        "model_class": "thermoroute.thermoroute.ThermoRoute",
        "stage": "09_usgs_experiment",
        "arm_id": member.arm_id,
        "seed": member.seed,
        "model_kwargs_intervention": dict(intervention),
        "train_config": dict(config["train_config"]),
        "delta_scale": config["delta_scale"],
        "device": "cpu",
        "eval_batch_size": config["eval_batch_size"],
        "station_sampling": "balanced",
        "selection_metric": "station_macro",
        "scope": "ablation_usgs",
        "feature_set": "USGS",
        "fixed_epoch_policy": {
            "maximum_epochs": config["train_config"]["max_epochs"],
            "early_stopping_patience": config["train_config"]["patience"],
        },
    }


def _authorization_document(
    *,
    root: Path,
    identity: RunIdentity,
    resolved_config: Mapping[str, Any],
    gate: ModelMatrixGate,
    interventions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(interventions) != tuple(STAGE09_CONTROLS):
        raise Stage09ParallelError("control intervention registry order changed")
    if identity.config_sha256 != sha256_json(resolved_config):
        raise Stage09ParallelError("resolved config differs from RunIdentity")
    matrix = []
    for member in expected_control_members():
        intervention = interventions.get(member.arm_id)
        if not isinstance(intervention, Mapping) or not intervention:
            raise Stage09ParallelError(
                f"control intervention is absent: {member.arm_id}"
            )
        scientific_config = _scientific_member_config(
            resolved_config, member, intervention
        )
        matrix.append(
            {
                **member.as_dict(),
                "scientific_config_sha256": sha256_json(scientific_config),
                "work_order": member.work_order_relative,
                "archive": member.archive_relative,
            }
        )
    stable = {
        "format": AUTHORIZATION_FORMAT,
        "status": "FROZEN_BEFORE_CONTROL_MEMBER_EXECUTION",
        "stage": "09_usgs_experiment",
        "run_identity": identity.as_dict(),
        "resolved_config": dict(resolved_config),
        "model_matrix": dict(gate.binding),
        "interventions": {
            arm_id: dict(interventions[arm_id]) for arm_id in STAGE09_CONTROLS
        },
        "matrix": matrix,
        "matrix_audit": {
            "arm_count": len(STAGE09_CONTROLS),
            "seed_count_per_arm": len(MODEL_MATRIX_SEEDS),
            "expected_member_count": CONTROL_MEMBER_COUNT,
            "arm_major_seed_minor_order": True,
            "same_seed_pairing_with_thermoroute": True,
            "missing_duplicate_or_extra_member_allowed": False,
        },
        "scientific_execution_contract": {
            "one_native_thread_per_member_process": True,
            "fixed_member_seed_config_and_epoch_policy": True,
            "members_share_no_mutable_scientific_state": True,
            "aggregation_uses_frozen_arm_major_seed_minor_order": True,
            "same_member_algorithm_as_serial_stage09": True,
            "worker_count_is_execution_only": True,
            "worker_count_enters_run_identity": False,
            "scheduling_order_enters_scientific_config": False,
        },
    }
    # Keep root out of the document: repository location is execution-only.
    del root
    return _with_self_hash(stable, "authorization_self_sha256")


def _work_order_document(
    *,
    root: Path,
    control_root: Path,
    authorization: Mapping[str, Any],
    member: ControlMember,
    intervention: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = authorization["resolved_config"]
    assert isinstance(resolved, Mapping)
    scientific = _scientific_member_config(resolved, member, intervention)
    authorization_relative = _relative_to_root(
        root, control_root / "authorization.json", label="authorization path"
    )
    stable = {
        "format": WORK_ORDER_FORMAT,
        "status": "AUTHORIZED_NOT_YET_COMMITTED",
        "stage": "09_usgs_experiment",
        "authorization": {
            "path": authorization_relative,
            "sha256": hashlib.sha256(
                _canonical_bytes(authorization) + b"\n"
            ).hexdigest(),
        },
        "run_identity": dict(authorization["run_identity"]),
        "model_matrix_contract_id": authorization["model_matrix"][
            "model_matrix_contract_id"
        ],
        "member": member.as_dict(),
        "scientific_member_config": scientific,
        "scientific_member_config_sha256": sha256_json(scientific),
        "artifact_contract": {
            "archive_commit": "same_filesystem_atomic_exclusive_rename",
            "archive_format": "uncompressed_deterministic_tar",
            "required_payloads": list(EXPECTED_MEMBER_PAYLOADS),
            "canonical_member_receipt_required": True,
            "unexpected_archive_member_allowed": False,
        },
    }
    return _with_self_hash(stable, "work_order_self_sha256")


def freeze_control_plan(
    *,
    root: str | Path,
    run_directory: str | Path,
    identity: RunIdentity,
    resolved_config: Mapping[str, Any],
    matrix_gate: ModelMatrixGate,
    interventions: Mapping[str, Mapping[str, Any]],
    publication_guard: Callable[[], object],
) -> tuple[Path, tuple[Path, ...]]:
    """Create or byte-verify the authority and exact 35 work orders."""
    repository = Path(root).resolve()
    expected_run = _run_directory(repository, identity.run_id)
    supplied_run = Path(os.path.abspath(os.fspath(run_directory)))
    if supplied_run != expected_run:
        raise Stage09ParallelError("control plan run directory is not canonical")
    manifest = _load_json(
        expected_run / "run.json",
        label="Stage-09 run manifest",
        repository_style=True,
    )
    if not isinstance(manifest, Mapping) or manifest.get("identity") != identity.as_dict():
        raise Stage09ParallelError("Stage-09 run manifest identity differs")
    _validate_formal_run_config(resolved_config)
    _validate_matrix_document(matrix_gate.amendment)
    publication_guard()
    control_root = _safe_directory(
        expected_run / CONTROL_PRECOMPUTE_DIRECTORY,
        parent=expected_run,
    )
    _mkdir_chain(control_root, "work_orders")
    _mkdir_chain(control_root, "members")
    _safe_directory(
        expected_run / ".stage09-control-member-locks",
        parent=expected_run,
    )
    _safe_directory(
        expected_run / ".stage09-control-member-staging",
        parent=expected_run,
    )
    _safe_directory(
        expected_run / CONTROL_RESUME_DIRECTORY,
        parent=expected_run,
    )
    authorization = _authorization_document(
        root=repository,
        identity=identity,
        resolved_config=resolved_config,
        gate=matrix_gate,
        interventions=interventions,
    )
    authorization_path = control_root / "authorization.json"
    _write_create_only(
        authorization_path,
        _canonical_bytes(authorization) + b"\n",
    )
    work_orders: list[Path] = []
    for member in expected_control_members():
        intervention = interventions[member.arm_id]
        document = _work_order_document(
            root=repository,
            control_root=control_root,
            authorization=authorization,
            member=member,
            intervention=intervention,
        )
        path = control_root / member.work_order_relative
        _mkdir_chain(control_root, str(Path(member.work_order_relative).parent))
        _mkdir_chain(control_root, str(Path(member.archive_relative).parent))
        _write_create_only(path, _canonical_bytes(document) + b"\n")
        work_orders.append(path)
    _assert_exact_plan_tree(control_root)
    publication_guard()
    return authorization_path, tuple(work_orders)


def _assert_exact_plan_tree(control_root: Path) -> None:
    expected_work_orders = {
        member.work_order_relative for member in expected_control_members()
    }
    work_root = control_root / "work_orders"
    actual_work_orders = {
        path.relative_to(control_root).as_posix()
        for path in work_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_work_orders != expected_work_orders:
        raise Stage09ParallelError(
            "Stage-09 work-order tree has a missing, duplicate, or extra file"
        )
    for path in work_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise Stage09ParallelError("Stage-09 work-order tree is unsafe")


def _validate_authorization(
    root: Path,
    path: Path,
) -> tuple[Mapping[str, Any], RunIdentity]:
    value = _load_json(path, label="Stage-09 control authorization")
    required = {
        "format",
        "status",
        "stage",
        "run_identity",
        "resolved_config",
        "model_matrix",
        "interventions",
        "matrix",
        "matrix_audit",
        "scientific_execution_contract",
    }
    document = _validate_self_hash(
        value,
        field="authorization_self_sha256",
        expected_keys=required,
        label="Stage-09 control authorization",
    )
    if (
        document.get("format") != AUTHORIZATION_FORMAT
        or document.get("status") != "FROZEN_BEFORE_CONTROL_MEMBER_EXECUTION"
        or document.get("stage") != "09_usgs_experiment"
    ):
        raise Stage09ParallelError("Stage-09 control authorization changed")
    identity = _identity_from_mapping(document["run_identity"])
    expected_path = _control_root(root, identity.run_id) / "authorization.json"
    if path != expected_path:
        raise Stage09ParallelError("authorization path is not canonical")
    config = _validate_formal_run_config(document["resolved_config"])
    if sha256_json(config) != identity.config_sha256:
        raise Stage09ParallelError("authorization config digest changed")
    binding = document.get("model_matrix")
    matrix_binding_fields = {
        "format",
        "amendment",
        "seal",
        "amendment_format",
        "amendment_status",
        "seal_format",
        "seal_status",
        "amendment_id",
        "amendment_document_commit",
        "model_matrix_contract_id",
    }
    if (
        not isinstance(binding, Mapping)
        or set(binding) != matrix_binding_fields
        or binding.get("format")
        != "thermoroute.stage09-control-model-matrix-binding.v1"
        or binding.get("amendment_format") != AMENDMENT_FORMAT
        or binding.get("amendment_status") != AMENDMENT_STATUS
        or binding.get("seal_format") != AMENDMENT_SEAL_FORMAT
        or binding.get("seal_status") != AMENDMENT_SEAL_STATUS
        or binding.get("amendment_id") != AMENDMENT_ID
        or type(binding.get("amendment_document_commit")) is not str
        or re.fullmatch(
            r"[0-9a-f]{40}", str(binding.get("amendment_document_commit"))
        )
        is None
        or type(binding.get("model_matrix_contract_id")) is not str
        or _DIGEST.fullmatch(str(binding.get("model_matrix_contract_id"))) is None
    ):
        raise Stage09ParallelError("authorization model-matrix binding changed")
    _assert_binding_bytes(root, binding["amendment"], label="matrix amendment")
    _assert_binding_bytes(root, binding["seal"], label="matrix amendment seal")
    interventions = document.get("interventions")
    if (
        not isinstance(interventions, Mapping)
        or set(interventions) != set(STAGE09_CONTROLS)
        or any(
            not isinstance(interventions.get(arm_id), Mapping)
            or not interventions[arm_id]
            for arm_id in STAGE09_CONTROLS
        )
    ):
        raise Stage09ParallelError("authorization intervention registry changed")
    expected_matrix = []
    for member in expected_control_members():
        expected_matrix.append(
            {
                **member.as_dict(),
                "scientific_config_sha256": sha256_json(
                    _scientific_member_config(
                        config,
                        member,
                        _intervention_from_authorized_config(
                            document, member.arm_id
                        ),
                    )
                ),
                "work_order": member.work_order_relative,
                "archive": member.archive_relative,
            }
        )
    if document.get("matrix") != expected_matrix:
        raise Stage09ParallelError("authorization member matrix changed")
    audit = document.get("matrix_audit")
    contract = document.get("scientific_execution_contract")
    if (
        audit
        != {
            "arm_count": 7,
            "seed_count_per_arm": 5,
            "expected_member_count": 35,
            "arm_major_seed_minor_order": True,
            "same_seed_pairing_with_thermoroute": True,
            "missing_duplicate_or_extra_member_allowed": False,
        }
        or not isinstance(contract, Mapping)
        or set(contract)
        != {
            "one_native_thread_per_member_process",
            "fixed_member_seed_config_and_epoch_policy",
            "members_share_no_mutable_scientific_state",
            "aggregation_uses_frozen_arm_major_seed_minor_order",
            "same_member_algorithm_as_serial_stage09",
            "worker_count_is_execution_only",
            "worker_count_enters_run_identity",
            "scheduling_order_enters_scientific_config",
        }
        or any(
            contract.get(field) is not True
            for field in (
                "one_native_thread_per_member_process",
                "fixed_member_seed_config_and_epoch_policy",
                "members_share_no_mutable_scientific_state",
                "aggregation_uses_frozen_arm_major_seed_minor_order",
                "same_member_algorithm_as_serial_stage09",
                "worker_count_is_execution_only",
            )
        )
        or contract.get("worker_count_enters_run_identity") is not False
        or contract.get("scheduling_order_enters_scientific_config") is not False
    ):
        raise Stage09ParallelError("authorization execution contract changed")
    return document, identity


def _intervention_from_authorized_config(
    authorization: Mapping[str, Any], arm_id: str
) -> Mapping[str, Any]:
    interventions = authorization.get("interventions")
    if not isinstance(interventions, Mapping):
        raise Stage09ParallelError("authorization intervention registry is malformed")
    intervention = interventions.get(arm_id)
    if not isinstance(intervention, Mapping) or not intervention:
        raise Stage09ParallelError(f"authorization lacks arm: {arm_id}")
    return intervention


def _member_from_document(value: object) -> ControlMember:
    required = {
        "member_id",
        "arm_index",
        "arm_id",
        "seed",
        "same_seed_reference",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Stage09ParallelError("control member schema changed")
    if type(value.get("arm_index")) is not int or type(value.get("seed")) is not int:
        raise Stage09ParallelError("control member index or seed is malformed")
    member = ControlMember(
        arm_index=int(value["arm_index"]),
        arm_id=str(value.get("arm_id", "")),
        seed=int(value["seed"]),
    )
    if member not in set(expected_control_members()) or value != member.as_dict():
        raise Stage09ParallelError("control member is outside the frozen matrix")
    return member


def validate_work_order_structure(
    *, root: str | Path, work_order_path: str | Path
) -> tuple[Mapping[str, Any], Mapping[str, Any], RunIdentity]:
    """Validate canonical bytes, paths, authority, config, and member identity."""
    repository = Path(root).resolve()
    relative = _relative_to_root(
        repository, Path(work_order_path), label="work-order path"
    )
    path = _safe_relative_file(repository, relative)
    value = _load_json(path, label="Stage-09 control work order")
    required = {
        "format",
        "status",
        "stage",
        "authorization",
        "run_identity",
        "model_matrix_contract_id",
        "member",
        "scientific_member_config",
        "scientific_member_config_sha256",
        "artifact_contract",
    }
    document = _validate_self_hash(
        value,
        field="work_order_self_sha256",
        expected_keys=required,
        label="Stage-09 control work order",
    )
    if (
        document.get("format") != WORK_ORDER_FORMAT
        or document.get("status") != "AUTHORIZED_NOT_YET_COMMITTED"
        or document.get("stage") != "09_usgs_experiment"
    ):
        raise Stage09ParallelError("Stage-09 work-order contract changed")
    identity = _identity_from_mapping(document["run_identity"])
    member = _member_from_document(document["member"])
    expected_path = _control_root(repository, identity.run_id) / (
        member.work_order_relative
    )
    if path != expected_path:
        raise Stage09ParallelError("work-order path/member binding changed")
    authorization_binding = document.get("authorization")
    if not isinstance(authorization_binding, Mapping) or set(
        authorization_binding
    ) != {"path", "sha256"}:
        raise Stage09ParallelError("work-order authorization binding changed")
    authorization_relative = authorization_binding.get("path")
    authorization_sha = authorization_binding.get("sha256")
    if (
        type(authorization_relative) is not str
        or type(authorization_sha) is not str
        or _DIGEST.fullmatch(authorization_sha) is None
    ):
        raise Stage09ParallelError("work-order authorization binding malformed")
    authorization_path = _safe_relative_file(repository, authorization_relative)
    if authorization_path != _control_root(
        repository, identity.run_id
    ) / "authorization.json":
        raise Stage09ParallelError("work order points to a noncanonical authority")
    authorization, authorized_identity = _validate_authorization(
        repository, authorization_path
    )
    if (
        _sha256_file(authorization_path) != authorization_sha
        or authorized_identity != identity
        or document["run_identity"] != authorization["run_identity"]
    ):
        raise Stage09ParallelError("work-order authorization bytes differ")
    model_matrix = authorization["model_matrix"]
    if (
        not isinstance(model_matrix, Mapping)
        or document.get("model_matrix_contract_id")
        != model_matrix.get("model_matrix_contract_id")
    ):
        raise Stage09ParallelError("work-order model-matrix identity changed")
    config = authorization["resolved_config"]
    assert isinstance(config, Mapping)
    intervention = _intervention_from_authorized_config(
        authorization, member.arm_id
    )
    expected_scientific = _scientific_member_config(
        config, member, intervention
    )
    if (
        document.get("scientific_member_config") != expected_scientific
        or document.get("scientific_member_config_sha256")
        != sha256_json(expected_scientific)
    ):
        raise Stage09ParallelError("work-order scientific member config changed")
    if document.get("artifact_contract") != {
        "archive_commit": "same_filesystem_atomic_exclusive_rename",
        "archive_format": "uncompressed_deterministic_tar",
        "required_payloads": list(EXPECTED_MEMBER_PAYLOADS),
        "canonical_member_receipt_required": True,
        "unexpected_archive_member_allowed": False,
    }:
        raise Stage09ParallelError("work-order artifact contract changed")
    authority_member = authorization["matrix"][
        member.arm_index * len(MODEL_MATRIX_SEEDS) + member.seed
    ]
    if (
        not isinstance(authority_member, Mapping)
        or authority_member.get("member_id") != member.member_id
        or authority_member.get("work_order") != member.work_order_relative
        or authority_member.get("archive") != member.archive_relative
        or authority_member.get("scientific_config_sha256")
        != document.get("scientific_member_config_sha256")
    ):
        raise Stage09ParallelError("work order differs from authority registry")
    return document, authorization, identity


def validate_live_work_order(
    *, root: str | Path, work_order_path: str | Path
) -> ValidatedWorkOrder:
    """Replay all formal gates before a worker may load data or mutate output."""
    repository = Path(root).resolve()
    document, authorization, identity = validate_work_order_structure(
        root=repository, work_order_path=work_order_path
    )
    assert_formal_numerical_policy()
    closure = resolve_development_input_closure(repository)
    closure.assert_unchanged()
    if closure.binding_digest != identity.input_closure_sha256:
        raise Stage09ParallelError("work-order input closure differs")
    config = authorization["resolved_config"]
    assert isinstance(config, Mapping)
    if config.get("input_closure_sha256") != closure.binding_digest:
        raise Stage09ParallelError("resolved config input closure differs")
    panel = repository / "data_usgs" / "panel_usgs_120v2.parquet"
    registry = repository / "data_usgs" / "station_registry_v1.csv"
    recomputed = resolve_run_identity(
        root=repository,
        panel=panel,
        registry=registry,
        config=config,
        input_closure_sha256=closure.binding_digest,
    )
    if recomputed != identity:
        raise Stage09ParallelError(
            "work-order source/data/runtime/config RunIdentity differs"
        )
    gate = validate_stage09_model_matrix_gate(repository)
    if dict(gate.binding) != authorization.get("model_matrix"):
        raise Stage09ParallelError("live model-matrix binding differs from authority")
    validated = ValidatedWorkOrder(
        root=repository,
        path=Path(work_order_path).resolve(),
        document=document,
        authorization=authorization,
        identity=identity,
        input_closure=closure,
        matrix_gate=gate,
    )
    validated.assert_unchanged()
    return validated


def _payload_registry(payload: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(payload.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(payload).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise Stage09ParallelError(f"member payload contains symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise Stage09ParallelError(
                f"member payload contains unsafe file: {relative}"
            )
        _normal_relative(relative, label="member artifact")
        records.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": info.st_size,
            }
        )
    required = set(EXPECTED_MEMBER_PAYLOADS)
    actual = {record["path"] for record in records}
    if actual != required:
        raise Stage09ParallelError(
            "member payload has a missing or unexpected artifact"
        )
    return records


def _validate_semantic_validation(
    value: object,
    *,
    member: ControlMember,
) -> dict[str, Any]:
    """Validate the source-bound callback's exact scientific audit result."""
    required = {
        "format",
        "model_id",
        "seed",
        "scope",
        "feature_set",
        "prediction_schema",
        "prediction_artifact_sha256",
        "prediction_sidecar_sha256",
        "bundle_metadata_sha256",
        "bundle_weights_sha256",
        "bundle_member_id",
        "checkpoint_payload_sha256",
        "checkpoint_sidecar_sha256",
        "checkpoint_best_epoch_index",
        "checkpoint_best_selection_metric_value",
        "same_seed_reference_prediction_sha256",
        "same_seed_reference_sidecar_sha256",
        "forecast_record_count",
        "forecast_key_sha256",
        "truth_float32_sha256",
        "prediction_sidecar_run_identity_exact",
        "bundle_metadata_identity_exact",
        "bundle_member_registry_exact",
        "bundle_intervention_exact",
        "bundle_weights_safely_loaded",
        "checkpoint_weights_safely_loaded",
        "bundle_matches_checkpoint_best_state_exact",
        "same_seed_forecast_keys_exact",
        "same_seed_split_exact",
        "same_seed_y_true_float32_exact",
        "full_prediction_replay_equivalence_proved",
        "full_prediction_replay_max_abs_difference",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Stage09ParallelError(
            "member semantic-validation schema is not exact"
        )
    digest_fields = {
        "prediction_artifact_sha256",
        "prediction_sidecar_sha256",
        "bundle_metadata_sha256",
        "bundle_weights_sha256",
        "checkpoint_payload_sha256",
        "checkpoint_sidecar_sha256",
        "same_seed_reference_prediction_sha256",
        "same_seed_reference_sidecar_sha256",
        "forecast_key_sha256",
        "truth_float32_sha256",
    }
    if (
        value.get("format") != SEMANTIC_VALIDATION_FORMAT
        or value.get("model_id") != member.arm_id
        or type(value.get("seed")) is not int
        or value.get("seed") != member.seed
        or value.get("scope") != "ablation_usgs"
        or value.get("feature_set") != "USGS"
        or value.get("prediction_schema") != "thermoroute.predictions.v1"
        or value.get("bundle_member_id") != f"seed{member.seed}"
        or type(value.get("checkpoint_best_epoch_index")) is not int
        or value["checkpoint_best_epoch_index"] < 0
        or isinstance(
            value.get("checkpoint_best_selection_metric_value"), bool
        )
        or not isinstance(
            value.get("checkpoint_best_selection_metric_value"), (int, float)
        )
        or not math.isfinite(
            float(value["checkpoint_best_selection_metric_value"])
        )
        or type(value.get("forecast_record_count")) is not int
        or value["forecast_record_count"] < 1
        or any(
            type(value.get(field)) is not str
            or _DIGEST.fullmatch(str(value.get(field))) is None
            for field in digest_fields
        )
        or any(
            value.get(field) is not True
            for field in (
                "prediction_sidecar_run_identity_exact",
                "bundle_metadata_identity_exact",
                "bundle_member_registry_exact",
                "bundle_intervention_exact",
                "bundle_weights_safely_loaded",
                "checkpoint_weights_safely_loaded",
                "bundle_matches_checkpoint_best_state_exact",
                "same_seed_forecast_keys_exact",
                "same_seed_split_exact",
                "same_seed_y_true_float32_exact",
                "full_prediction_replay_equivalence_proved",
            )
        )
        or isinstance(
            value.get("full_prediction_replay_max_abs_difference"), bool
        )
        or not isinstance(
            value.get("full_prediction_replay_max_abs_difference"), (int, float)
        )
        or float(value["full_prediction_replay_max_abs_difference"]) != 0.0
    ):
        raise Stage09ParallelError("member semantic validation failed closed")
    # Make the callback result independent of custom Mapping subclasses and
    # prove it can be represented by the receipt's canonical JSON domain.
    result = dict(value)
    _canonical_bytes(result)
    return result


def _run_semantic_validator(
    *,
    payload: Path,
    validated: ValidatedWorkOrder,
    semantic_validator: Callable[
        [Path, ValidatedWorkOrder], Mapping[str, Any]
    ],
) -> dict[str, Any]:
    validated.assert_unchanged()
    try:
        value = semantic_validator(payload, validated)
    except Stage09ParallelError:
        raise
    except Exception as exc:
        raise Stage09ParallelError(
            f"member scientific semantic validation failed: {validated.member.member_id}"
        ) from exc
    result = _validate_semantic_validation(value, member=validated.member)
    validated.assert_unchanged()
    return result


def _member_receipt(
    *,
    work_order: Mapping[str, Any],
    work_order_path: Path,
    root: Path,
    artifacts: Sequence[Mapping[str, Any]],
    result_metadata: Mapping[str, Any],
    semantic_validation: Mapping[str, Any],
) -> dict[str, Any]:
    best_epoch = result_metadata.get("best_epoch_index")
    best_value = result_metadata.get("best_selection_metric_value")
    if (
        type(best_epoch) is not int
        or best_epoch < 0
        or isinstance(best_value, bool)
        or not isinstance(best_value, (int, float))
        or not math.isfinite(float(best_value))
    ):
        raise Stage09ParallelError("member result metadata is malformed")
    checkpoint_value = semantic_validation.get(
        "checkpoint_best_selection_metric_value"
    )
    if (
        semantic_validation.get("checkpoint_best_epoch_index") != best_epoch
        or isinstance(checkpoint_value, bool)
        or not isinstance(checkpoint_value, (int, float))
        or float(checkpoint_value) != float(best_value)
    ):
        raise Stage09ParallelError(
            "member result metadata differs from its safe checkpoint replay"
        )
    member = _member_from_document(work_order["member"])
    stable = {
        "format": MEMBER_RECEIPT_FORMAT,
        "status": "COMPLETE_AND_CREATE_ONLY",
        "stage": "09_usgs_experiment",
        "work_order": {
            "path": _relative_to_root(
                root, work_order_path, label="member receipt work-order path"
            ),
            "sha256": _sha256_file(work_order_path),
        },
        "run_identity": dict(work_order["run_identity"]),
        "model_matrix_contract_id": work_order["model_matrix_contract_id"],
        "member": member.as_dict(),
        "scientific_member_config": dict(
            work_order["scientific_member_config"]
        ),
        "scientific_member_config_sha256": work_order[
            "scientific_member_config_sha256"
        ],
        "artifacts": [dict(item) for item in artifacts],
        "semantic_validation": dict(semantic_validation),
        "result_metadata": {
            "best_epoch_index": best_epoch,
            "best_selection_metric_value": float(best_value),
        },
        "execution_attestation": {
            "native_threads_per_worker": 1,
            "member_process_isolated": True,
            "worker_count_recorded_in_scientific_identity": False,
            "publication_is_single_archive_commit": True,
        },
    }
    return _with_self_hash(stable, "receipt_self_sha256")


def _write_deterministic_tar(
    *, payload: Path, receipt: Mapping[str, Any], destination: Path
) -> None:
    receipt_bytes = _canonical_bytes(receipt) + b"\n"
    files = [
        path
        for path in payload.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    names = [path.relative_to(payload).as_posix() for path in files]
    if "receipt.json" in names:
        raise Stage09ParallelError("payload may not precreate receipt.json")
    entries: list[tuple[str, Path | None, bytes | None]] = [
        (name, path, None) for name, path in zip(names, files, strict=True)
    ]
    entries.append(("receipt.json", None, receipt_bytes))
    entries.sort(key=lambda item: item[0])
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as raw:
            with tarfile.open(
                fileobj=raw,
                mode="w",
                format=tarfile.GNU_FORMAT,
            ) as archive:
                for name, path, supplied_bytes in entries:
                    _normal_relative(name, label="archive member")
                    size = (
                        len(supplied_bytes)
                        if supplied_bytes is not None
                        else path.stat().st_size if path is not None else 0
                    )
                    info = tarfile.TarInfo(name=name)
                    info.size = size
                    info.mode = 0o600
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    if supplied_bytes is not None:
                        archive.addfile(info, io.BytesIO(supplied_bytes))
                    elif path is not None:
                        with path.open("rb") as source:
                            archive.addfile(info, source)
            raw.flush()
            os.fsync(raw.fileno())
    finally:
        os.close(descriptor)


def _archive_member_digest(
    archive: tarfile.TarFile, info: tarfile.TarInfo
) -> tuple[str, int, bytes | None]:
    source = archive.extractfile(info)
    if source is None:
        raise Stage09ParallelError(f"cannot read archive member: {info.name}")
    digest = hashlib.sha256()
    total = 0
    captured = bytearray() if info.name == "receipt.json" else None
    while True:
        block = source.read(1024 * 1024)
        if not block:
            break
        total += len(block)
        if captured is not None:
            if total > _MAX_RECEIPT_BYTES:
                raise Stage09ParallelError("member receipt exceeds size cap")
            captured.extend(block)
        digest.update(block)
    if total != info.size:
        raise Stage09ParallelError(f"archive member size differs: {info.name}")
    return digest.hexdigest(), total, bytes(captured) if captured is not None else None


def _assert_canonical_tar_layout(
    path: Path,
    members: Sequence[tarfile.TarInfo],
    *,
    logical_end: int,
) -> None:
    """Reject changed headers, inter-member padding, or trailing tar bytes."""
    expected_size = (
        (logical_end + (2 * tarfile.BLOCKSIZE) + tarfile.RECORDSIZE - 1)
        // tarfile.RECORDSIZE
    ) * tarfile.RECORDSIZE
    if path.stat().st_size != expected_size:
        raise Stage09ParallelError("member archive has noncanonical trailing length")
    with path.open("rb") as raw:
        for item in members:
            expected_header = item.tobuf(
                format=tarfile.GNU_FORMAT,
                encoding="utf-8",
                errors="surrogateescape",
            )
            header_length = item.offset_data - item.offset
            raw.seek(item.offset)
            if (
                len(expected_header) != header_length
                or raw.read(header_length) != expected_header
            ):
                raise Stage09ParallelError(
                    f"archive member header changed: {item.name}"
                )
            padding_start = item.offset_data + item.size
            padding_end = (
                (padding_start + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
            ) * tarfile.BLOCKSIZE
            raw.seek(padding_start)
            if any(raw.read(padding_end - padding_start)):
                raise Stage09ParallelError(
                    f"archive member padding changed: {item.name}"
                )
        raw.seek(logical_end)
        if any(raw.read(expected_size - logical_end)):
            raise Stage09ParallelError("member archive terminator padding changed")


def validate_member_archive(
    *,
    root: str | Path,
    work_order_path: str | Path,
    archive_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Strictly replay a committed member without extracting or trusting it."""
    repository = Path(root).resolve()
    work_order, _authorization, identity = validate_work_order_structure(
        root=repository, work_order_path=work_order_path
    )
    member = _member_from_document(work_order["member"])
    expected_archive = _control_root(repository, identity.run_id) / (
        member.archive_relative
    )
    supplied = expected_archive if archive_path is None else Path(archive_path)
    supplied = Path(os.path.abspath(os.fspath(supplied)))
    if supplied != expected_archive:
        raise Stage09ParallelError("member archive path is not canonical")
    try:
        info = supplied.lstat()
    except FileNotFoundError as exc:
        raise Stage09ParallelError("member archive is absent") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size > _MAX_ARCHIVE_BYTES
    ):
        raise Stage09ParallelError("member archive is unsafe or exceeds size cap")
    try:
        with tarfile.open(supplied, mode="r:") as archive:
            members = archive.getmembers()
            if not members or len(members) > _MAX_ARCHIVE_MEMBERS:
                raise Stage09ParallelError("member archive count is invalid")
            names = [item.name for item in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise Stage09ParallelError(
                    "member archive order or uniqueness changed"
                )
            digests: dict[str, tuple[str, int]] = {}
            receipt_bytes: bytes | None = None
            for item in members:
                _normal_relative(item.name, label="archive member")
                if (
                    not item.isreg()
                    or item.mode != 0o600
                    or item.uid != 0
                    or item.gid != 0
                    or item.uname != ""
                    or item.gname != ""
                    or item.mtime != 0
                ):
                    raise Stage09ParallelError(
                        f"archive member metadata changed: {item.name}"
                    )
                digest, size, captured = _archive_member_digest(archive, item)
                digests[item.name] = (digest, size)
                if captured is not None:
                    receipt_bytes = captured
            logical_end = archive.offset
    except (tarfile.TarError, OSError) as exc:
        raise Stage09ParallelError("member archive is malformed") from exc
    _assert_canonical_tar_layout(
        supplied,
        members,
        logical_end=logical_end,
    )
    if receipt_bytes is None or "receipt.json" not in digests:
        raise Stage09ParallelError("member archive lacks its commit receipt")
    receipt_value = _load_canonical_json_bytes(
        receipt_bytes,
        label="Stage-09 control member receipt",
    )
    required_receipt = {
        "format",
        "status",
        "stage",
        "work_order",
        "run_identity",
        "model_matrix_contract_id",
        "member",
        "scientific_member_config",
        "scientific_member_config_sha256",
        "artifacts",
        "semantic_validation",
        "result_metadata",
        "execution_attestation",
    }
    receipt = _validate_self_hash(
        receipt_value,
        field="receipt_self_sha256",
        expected_keys=required_receipt,
        label="Stage-09 control member receipt",
    )
    if (
        receipt.get("format") != MEMBER_RECEIPT_FORMAT
        or receipt.get("status") != "COMPLETE_AND_CREATE_ONLY"
        or receipt.get("stage") != "09_usgs_experiment"
        or receipt.get("work_order")
        != {
            "path": _relative_to_root(
                repository,
                Path(work_order_path),
                label="validated work-order path",
            ),
            "sha256": _sha256_file(Path(work_order_path)),
        }
        or receipt.get("run_identity") != work_order.get("run_identity")
        or receipt.get("model_matrix_contract_id")
        != work_order.get("model_matrix_contract_id")
        or receipt.get("member") != member.as_dict()
        or receipt.get("scientific_member_config")
        != work_order.get("scientific_member_config")
        or receipt.get("scientific_member_config_sha256")
        != work_order.get("scientific_member_config_sha256")
        or receipt.get("execution_attestation")
        != {
            "native_threads_per_worker": 1,
            "member_process_isolated": True,
            "worker_count_recorded_in_scientific_identity": False,
            "publication_is_single_archive_commit": True,
        }
    ):
        raise Stage09ParallelError("member receipt contract changed")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Stage09ParallelError("member artifact registry is malformed")
    expected_names = {"receipt.json"}
    previous = ""
    for record in artifacts:
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise Stage09ParallelError("member artifact descriptor changed")
        name = record.get("path")
        record_digest: Any = record.get("sha256")
        byte_count = record.get("bytes")
        if (
            type(name) is not str
            or _normal_relative(name, label="receipt artifact") != name
            or name <= previous
            or type(record_digest) is not str
            or _DIGEST.fullmatch(record_digest) is None
            or type(byte_count) is not int
            or byte_count < 0
            or digests.get(name) != (record_digest, byte_count)
        ):
            raise Stage09ParallelError("member artifact bytes or order changed")
        previous = name
        expected_names.add(name)
    if set(digests) != expected_names:
        raise Stage09ParallelError("member archive contains an unexpected file")
    required_payloads = work_order["artifact_contract"]["required_payloads"]
    if not set(required_payloads) <= (expected_names - {"receipt.json"}):
        raise Stage09ParallelError("member receipt omits a required payload")
    semantic = receipt.get("semantic_validation")
    _validate_semantic_validation(semantic, member=member)
    result = receipt.get("result_metadata")
    if (
        not isinstance(result, Mapping)
        or set(result)
        != {"best_epoch_index", "best_selection_metric_value"}
        or type(result.get("best_epoch_index")) is not int
        or result["best_epoch_index"] < 0
        or isinstance(result.get("best_selection_metric_value"), bool)
        or not isinstance(
            result.get("best_selection_metric_value"), (int, float)
        )
        or not math.isfinite(float(result["best_selection_metric_value"]))
    ):
        raise Stage09ParallelError("member result metadata changed")
    return receipt


def publish_control_member(
    *,
    validated: ValidatedWorkOrder,
    producer: Callable[[Path], Mapping[str, Any]],
    semantic_validator: Callable[
        [Path, ValidatedWorkOrder], Mapping[str, Any]
    ],
) -> tuple[Path, bool]:
    """Train, semantically verify, then atomically publish one archive."""
    member = validated.member
    control_root = _control_root(validated.root, validated.identity.run_id)
    archive_path = control_root / member.archive_relative
    locks_root = _safe_directory(
        validated.run_directory / ".stage09-control-member-locks",
        parent=validated.run_directory,
    )
    staging_root = _safe_directory(
        validated.run_directory / ".stage09-control-member-staging",
        parent=validated.run_directory,
    )
    lock_path = locks_root / f"{member.member_id}.lock"
    with advisory_file_lock(lock_path, exclusive=True):
        _prepare_member_resume_checkpoint(validated)
        if os.path.lexists(archive_path):
            # Reuse is intentionally strict: any partial path or changed byte
            # blocks the retry and is never replaced.
            receipt = validate_member_archive(
                root=validated.root,
                work_order_path=validated.path,
                archive_path=archive_path,
            )
            with extracted_member_payload(
                root=validated.root,
                work_order_path=validated.path,
            ) as existing_payload:
                semantic = _run_semantic_validator(
                    payload=existing_payload,
                    validated=validated,
                    semantic_validator=semantic_validator,
                )
            if semantic != receipt.get("semantic_validation"):
                raise Stage09ParallelError(
                    "reused member semantic evidence changed"
                )
            validated.assert_unchanged()
            _cleanup_member_resume_checkpoint(validated)
            return archive_path, True
        staging = Path(
            tempfile.mkdtemp(prefix=f".{member.member_id}.", dir=staging_root)
        )
        payload = staging / "payload"
        payload.mkdir(mode=0o700)
        tar_path = staging / "candidate.member.tar"
        try:
            validated.assert_unchanged()
            result_metadata = producer(payload)
            validated.assert_unchanged()
            artifacts = _payload_registry(payload)
            semantic_validation = _run_semantic_validator(
                payload=payload,
                validated=validated,
                semantic_validator=semantic_validator,
            )
            receipt = _member_receipt(
                work_order=validated.document,
                work_order_path=validated.path,
                root=validated.root,
                artifacts=artifacts,
                result_metadata=result_metadata,
                semantic_validation=semantic_validation,
            )
            _write_deterministic_tar(
                payload=payload,
                receipt=receipt,
                destination=tar_path,
            )
            validated.assert_unchanged()
            _mkdir_chain(
                control_root,
                str(Path(member.archive_relative).parent),
            )
            _atomic_exclusive_rename(tar_path, archive_path)
            committed_receipt = validate_member_archive(
                root=validated.root,
                work_order_path=validated.path,
                archive_path=archive_path,
            )
            with extracted_member_payload(
                root=validated.root,
                work_order_path=validated.path,
            ) as committed_payload:
                committed_semantic = _run_semantic_validator(
                    payload=committed_payload,
                    validated=validated,
                    semantic_validator=semantic_validator,
                )
            if (
                committed_semantic != semantic_validation
                or committed_receipt.get("semantic_validation")
                != semantic_validation
            ):
                raise Stage09ParallelError(
                    "committed member semantic evidence changed"
                )
            validated.assert_unchanged()
            _cleanup_member_resume_checkpoint(validated)
            return archive_path, False
        finally:
            shutil.rmtree(staging, ignore_errors=True)


@contextmanager
def extracted_member_payload(
    *, root: str | Path, work_order_path: str | Path
) -> Iterator[Path]:
    """Validate, then manually extract one archive into an owner-private temp dir."""
    repository = Path(root).resolve()
    work_order, _authorization, identity = validate_work_order_structure(
        root=repository, work_order_path=work_order_path
    )
    member = _member_from_document(work_order["member"])
    archive_path = _control_root(repository, identity.run_id) / (
        member.archive_relative
    )
    receipt = validate_member_archive(
        root=repository,
        work_order_path=work_order_path,
        archive_path=archive_path,
    )
    expected = {
        record["path"]: (record["sha256"], record["bytes"])
        for record in receipt["artifacts"]
    }
    with tempfile.TemporaryDirectory(prefix="thermoroute-stage09-member-") as temp:
        destination = Path(temp).resolve()
        with tarfile.open(archive_path, mode="r:") as archive:
            for info in archive.getmembers():
                if info.name == "receipt.json":
                    continue
                if info.name not in expected:
                    raise Stage09ParallelError(
                        "validated archive extraction registry changed"
                    )
                target = destination / info.name
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(info)
                if source is None:
                    raise Stage09ParallelError(
                        f"cannot extract member artifact: {info.name}"
                    )
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(target, flags, 0o600)
                digest = hashlib.sha256()
                byte_count = 0
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as handle:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            handle.write(block)
                            digest.update(block)
                            byte_count += len(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    os.close(descriptor)
                if (digest.hexdigest(), byte_count) != expected[info.name]:
                    raise Stage09ParallelError(
                        f"extracted member artifact changed: {info.name}"
                    )
        yield destination


def execute_work_orders_bounded(
    work_orders: Sequence[Path],
    *,
    max_workers: int,
    launch: Callable[[Path], int],
    publication_guard: Callable[[], object],
    terminate: Callable[[], object] | None = None,
) -> None:
    """Launch the exact registry with bounded, fail-fast execution.

    Only ``max_workers`` members are submitted at once.  A first failure stops
    replenishing the queue, cancels work that has not started, and asks the
    caller-owned launcher to terminate its live processes before joining the
    worker threads.  This prevents a Ctrl-C or one failed member from silently
    launching the remainder of the 35-member matrix.
    """
    if type(max_workers) is not int or not 1 <= max_workers <= MAX_CONTROL_WORKERS:
        raise Stage09ParallelError(
            f"control workers must be in [1, {MAX_CONTROL_WORKERS}]"
        )
    if len(work_orders) != CONTROL_MEMBER_COUNT or len(set(work_orders)) != len(
        work_orders
    ):
        raise Stage09ParallelError("scheduler did not receive exact 35 work orders")
    publication_guard()
    failures: list[tuple[str, object]] = []
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="stage09-control-launch",
    )
    iterator = iter(work_orders)
    future_to_path: dict[Any, Path] = {}

    def replenish() -> None:
        while len(future_to_path) < max_workers:
            try:
                path = next(iterator)
            except StopIteration:
                return
            future_to_path[executor.submit(launch, path)] = path

    def stop_live_processes() -> None:
        if terminate is not None:
            terminate()

    try:
        replenish()
        while future_to_path:
            completed, _pending = wait(
                tuple(future_to_path), return_when=FIRST_COMPLETED
            )
            for future in completed:
                path = future_to_path.pop(future)
                try:
                    return_code = future.result()
                except BaseException as exc:
                    failures.append((path.as_posix(), exc))
                    continue
                if type(return_code) is not int or return_code != 0:
                    failures.append((path.as_posix(), return_code))
            if failures:
                stop_live_processes()
                for future in future_to_path:
                    future.cancel()
                break
            replenish()
    except BaseException:
        try:
            stop_live_processes()
        finally:
            for future in future_to_path:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True, cancel_futures=True)
    if failures:
        summary = "; ".join(
            f"{path}={value!r}" for path, value in sorted(failures)
        )
        raise Stage09ParallelError(f"Stage-09 control worker failed: {summary}")
    publication_guard()


def _actual_member_archives(control_root: Path) -> set[str]:
    members_root = control_root / "members"
    actual: set[str] = set()
    for path in members_root.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise Stage09ParallelError("Stage-09 member tree contains unsafe entry")
        if path.is_file():
            actual.add(path.relative_to(control_root).as_posix())
    return actual


def _assert_execution_namespaces_quiescent(run_directory: Path) -> None:
    """Reject hidden interrupted state before the coordinator can commit.

    Local Git excludes may hide ``outputs/runs``.  Scientific completeness must
    therefore be proved from the filesystem itself: staging and resume roots
    are empty, while the execution-only lock root contains exactly one safe
    lock for every frozen member and no other object.
    """
    expected_locks = {
        f"{member.member_id}.lock" for member in expected_control_members()
    }
    namespaces = {
        "locks": (
            run_directory / ".stage09-control-member-locks",
            expected_locks,
        ),
        "staging": (
            run_directory / ".stage09-control-member-staging",
            set(),
        ),
        "resume": (
            run_directory / CONTROL_RESUME_DIRECTORY,
            set(),
        ),
    }
    for label, (directory, expected_names) in namespaces.items():
        try:
            directory_info = directory.lstat()
        except OSError as exc:
            raise Stage09ParallelError(
                f"Stage-09 {label} namespace is absent"
            ) from exc
        owner_matches = (
            not hasattr(os, "getuid") or directory_info.st_uid == os.getuid()
        )
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or stat.S_ISLNK(directory_info.st_mode)
            or not owner_matches
            or stat.S_IMODE(directory_info.st_mode) & 0o077
        ):
            raise Stage09ParallelError(
                f"Stage-09 {label} namespace is unsafe"
            )
        entries = tuple(directory.iterdir())
        if {entry.name for entry in entries} != expected_names:
            raise Stage09ParallelError(
                f"Stage-09 {label} namespace is not quiescent and exact"
            )
        for entry in entries:
            info = entry.lstat()
            entry_owner_matches = (
                not hasattr(os, "getuid") or info.st_uid == os.getuid()
            )
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_nlink != 1
                or not entry_owner_matches
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise Stage09ParallelError(
                    f"Stage-09 {label} namespace contains an unsafe entry"
                )


def _thermoroute_reference_registry(
    *,
    root: Path,
    identity: RunIdentity,
    references: Mapping[int, Mapping[str, Path]],
) -> list[dict[str, Any]]:
    """Bind the five already-validated full-model members used for pairing."""
    if set(references) != set(MODEL_MATRIX_SEEDS):
        raise Stage09ParallelError(
            "ThermoRoute same-seed reference registry is not exact 0--4"
        )
    registry: list[dict[str, Any]] = []
    for seed in MODEL_MATRIX_SEEDS:
        descriptor = references.get(int(seed))
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "prediction",
            "bundle",
        }:
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} reference descriptor changed"
            )
        prediction = Path(descriptor["prediction"])
        expected_prediction = (
            _run_directory(root, identity.run_id)
            / "predictions"
            / f"thermoroute_seed{seed}.parquet"
        )
        if Path(os.path.abspath(os.fspath(prediction))) != expected_prediction:
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} prediction path changed"
            )
        prediction_relative = _relative_to_root(
            root, prediction, label="ThermoRoute reference prediction"
        )
        prediction = _safe_relative_file(root, prediction_relative)
        sidecar = prediction.with_name(prediction.name + ".meta.json")
        sidecar_relative = _relative_to_root(
            root, sidecar, label="ThermoRoute reference sidecar"
        )
        sidecar = _safe_relative_file(root, sidecar_relative)
        sidecar_document = _load_json(
            sidecar,
            label=f"ThermoRoute seed{seed} prediction sidecar",
            repository_style=True,
        )
        if (
            not isinstance(sidecar_document, Mapping)
            or sidecar_document.get("run") != identity.as_dict()
            or sidecar_document.get("artifact") != prediction.name
            or sidecar_document.get("artifact_sha256") != _sha256_file(prediction)
            or sidecar_document.get("artifact_bytes") != prediction.stat().st_size
            or sidecar_document.get("kind")
            != "thermoroute_seed_predictions"
        ):
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} prediction lineage changed"
            )

        bundle = Path(os.path.abspath(os.fspath(descriptor["bundle"])))
        expected_bundle = (
            _run_directory(root, identity.run_id)
            / "member_bundles"
            / f"seed{seed}"
        )
        if bundle != expected_bundle or bundle.is_symlink() or not bundle.is_dir():
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} bundle path changed"
            )
        bundle_files = {
            path.relative_to(bundle).as_posix(): path
            for path in bundle.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if set(bundle_files) != {"metadata.json", "weights.pt"} or any(
            path.is_symlink() or (not path.is_dir() and not path.is_file())
            for path in bundle.rglob("*")
        ):
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} bundle tree changed"
            )
        metadata = _load_json(
            bundle_files["metadata.json"],
            label=f"ThermoRoute seed{seed} bundle metadata",
            repository_style=True,
        )
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("run_id") != identity.run_id
            or metadata.get("source_sha256") != identity.source_sha256
            or metadata.get("panel_sha256") != identity.panel_sha256
            or metadata.get("registry_sha256") != identity.registry_sha256
            or metadata.get("runtime_sha256") != identity.runtime_sha256
            or metadata.get("input_closure_sha256")
            != identity.input_closure_sha256
            or metadata.get("members") != [f"seed{seed}"]
            or metadata.get("member_count") != 1
            or metadata.get("weights_sha256")
            != _sha256_file(bundle_files["weights.pt"])
        ):
            raise Stage09ParallelError(
                f"ThermoRoute seed{seed} bundle lineage changed"
            )
        bundle_inventory = [
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for relative, path in sorted(bundle_files.items())
        ]
        registry.append(
            {
                "model_id": "ThermoRoute",
                "seed": int(seed),
                "prediction": {
                    "path": prediction_relative,
                    "sha256": _sha256_file(prediction),
                    "bytes": prediction.stat().st_size,
                },
                "prediction_sidecar": {
                    "path": sidecar_relative,
                    "sha256": _sha256_file(sidecar),
                    "bytes": sidecar.stat().st_size,
                },
                "bundle": {
                    "path": _relative_to_root(
                        root, bundle, label="ThermoRoute reference bundle"
                    ),
                    "inventory": bundle_inventory,
                    "inventory_sha256": sha256_json(bundle_inventory),
                },
            }
        )
    return registry


def freeze_control_matrix_receipt(
    *,
    root: str | Path,
    authorization_path: str | Path,
    work_orders: Sequence[Path],
    thermoroute_references: Mapping[int, Mapping[str, Path]],
    semantic_validator: Callable[
        [Path, ValidatedWorkOrder], Mapping[str, Any]
    ],
    publication_guard: Callable[[], object],
) -> Path:
    """Semantically replay exact 35 archives, then freeze their receipt."""
    repository = Path(root).resolve()
    authorization_relative = _relative_to_root(
        repository, Path(authorization_path), label="authorization path"
    )
    canonical_authorization_path = _safe_relative_file(
        repository, authorization_relative
    )
    authorization, identity = _validate_authorization(
        repository, canonical_authorization_path
    )
    expected_work_orders = tuple(
        _control_root(repository, identity.run_id) / member.work_order_relative
        for member in expected_control_members()
    )
    if tuple(work_orders) != expected_work_orders:
        raise Stage09ParallelError(
            "coordinator work-order registry/order is not exact 35"
        )
    control_root = _control_root(repository, identity.run_id)
    _assert_exact_plan_tree(control_root)
    expected_archives = {
        member.archive_relative for member in expected_control_members()
    }
    actual_archives = _actual_member_archives(control_root)
    if actual_archives != expected_archives:
        missing = sorted(expected_archives - actual_archives)
        extra = sorted(actual_archives - expected_archives)
        raise Stage09ParallelError(
            "Stage-09 member matrix is incomplete or has extras: "
            f"missing={missing}; extra={extra}"
        )
    _assert_execution_namespaces_quiescent(_run_directory(repository, identity.run_id))
    publication_guard()
    reference_registry = _thermoroute_reference_registry(
        root=repository,
        identity=identity,
        references=thermoroute_references,
    )
    reference_by_seed = {
        int(item["seed"]): item for item in reference_registry
    }
    registry = []
    matrix_binding = authorization["model_matrix"]
    assert isinstance(matrix_binding, Mapping)
    coordinator_gate = ModelMatrixGate(
        amendment={},
        seal={},
        binding=matrix_binding,
        contract_id=str(matrix_binding["model_matrix_contract_id"]),
    )
    closure = _PublicationGuardClosure(publication_guard)
    seen: set[tuple[str, int]] = set()
    seeds_by_arm: dict[str, list[int]] = {arm: [] for arm in STAGE09_CONTROLS}
    pairing_by_seed: dict[int, tuple[int, str, str]] = {}
    for member, work_order_path in zip(
        expected_control_members(), expected_work_orders, strict=True
    ):
        receipt = validate_member_archive(
            root=repository,
            work_order_path=work_order_path,
        )
        work_order, member_authorization, member_identity = (
            validate_work_order_structure(
                root=repository,
                work_order_path=work_order_path,
            )
        )
        if member_identity != identity or member_authorization != authorization:
            raise Stage09ParallelError(
                "coordinator member authority or RunIdentity changed"
            )
        semantic_view = ValidatedWorkOrder(
            root=repository,
            path=work_order_path,
            document=work_order,
            authorization=member_authorization,
            identity=member_identity,
            input_closure=closure,
            matrix_gate=coordinator_gate,
        )
        with extracted_member_payload(
            root=repository,
            work_order_path=work_order_path,
        ) as payload:
            semantic = _run_semantic_validator(
                payload=payload,
                validated=semantic_view,
                semantic_validator=semantic_validator,
            )
        if semantic != receipt.get("semantic_validation"):
            raise Stage09ParallelError(
                "coordinator member semantic evidence changed"
            )
        reference = reference_by_seed.get(member.seed)
        if (
            not isinstance(reference, Mapping)
            or semantic.get("same_seed_reference_prediction_sha256")
            != reference["prediction"]["sha256"]
            or semantic.get("same_seed_reference_sidecar_sha256")
            != reference["prediction_sidecar"]["sha256"]
        ):
            raise Stage09ParallelError(
                "coordinator semantic pairing refers to another reference"
            )
        pairing = (
            int(semantic["forecast_record_count"]),
            str(semantic["forecast_key_sha256"]),
            str(semantic["truth_float32_sha256"]),
        )
        prior_pairing = pairing_by_seed.setdefault(member.seed, pairing)
        if pairing != prior_pairing:
            raise Stage09ParallelError(
                "same-seed control members disagree on keys or truth"
            )
        observed = _member_from_document(receipt["member"])
        key = (observed.arm_id, observed.seed)
        if observed != member or key in seen:
            raise Stage09ParallelError(
                "Stage-09 member matrix has duplicate or reordered member"
            )
        seen.add(key)
        seeds_by_arm[member.arm_id].append(member.seed)
        archive_path = control_root / member.archive_relative
        registry.append(
            {
                **member.as_dict(),
                "work_order": {
                    "path": _relative_to_root(
                        repository, work_order_path, label="coordinator work order"
                    ),
                    "sha256": _sha256_file(work_order_path),
                },
                "archive": {
                    "path": _relative_to_root(
                        repository, archive_path, label="coordinator member archive"
                    ),
                    "sha256": _sha256_file(archive_path),
                    "bytes": archive_path.stat().st_size,
                },
                "member_receipt_sha256": receipt["receipt_self_sha256"],
                "scientific_member_config_sha256": receipt[
                    "scientific_member_config_sha256"
                ],
                "semantic_validation_sha256": sha256_json(semantic),
            }
        )
    if (
        len(seen) != CONTROL_MEMBER_COUNT
        or any(
            tuple(seeds_by_arm[arm]) != tuple(MODEL_MATRIX_SEEDS)
            for arm in STAGE09_CONTROLS
        )
    ):
        raise Stage09ParallelError("Stage-09 exact same-seed matrix proof failed")
    if set(pairing_by_seed) != set(MODEL_MATRIX_SEEDS):
        raise Stage09ParallelError("Stage-09 reference pairing registry is incomplete")
    for reference in reference_registry:
        count, key_sha256, truth_sha256 = pairing_by_seed[int(reference["seed"])]
        reference["forecast_record_count"] = count
        reference["forecast_key_sha256"] = key_sha256
        reference["truth_float32_sha256"] = truth_sha256
    stable = {
        "format": COORDINATOR_RECEIPT_FORMAT,
        "status": "EXACT_35_MEMBER_MATRIX_COMPLETE",
        "stage": "09_usgs_experiment",
        "authorization": {
            "path": authorization_relative,
            "sha256": _sha256_file(canonical_authorization_path),
            "authorization_self_sha256": authorization[
                "authorization_self_sha256"
            ],
        },
        "run_identity": identity.as_dict(),
        "model_matrix": dict(authorization["model_matrix"]),
        "same_seed_thermoroute_references": reference_registry,
        "member_registry": registry,
        "matrix_audit": {
            "expected_member_count": CONTROL_MEMBER_COUNT,
            "observed_member_count": len(registry),
            "duplicate_member_count": 0,
            "missing_member_count": 0,
            "extra_member_count": 0,
            "arms": list(STAGE09_CONTROLS),
            "seeds_per_arm": list(MODEL_MATRIX_SEEDS),
            "same_seed_reference_artifacts_bound": True,
            "same_seed_forecast_keys_and_y_true_exactly_verified": True,
            "full_prediction_replay_equivalence_proved": True,
        },
        "serial_semantics_contract": {
            "same_member_algorithm_config_seed_and_epoch_policy": True,
            "member_results_sorted_before_aggregation": True,
            "ensemble_uses_equal_weight_across_exact_five_seeds": True,
            "parallelism_changes_only_wall_clock_scheduling": True,
            "parallelism_enters_scientific_identity": False,
            "best_seed_selection_allowed": False,
        },
    }
    document = _with_self_hash(stable, "receipt_self_sha256")
    destination = control_root / "coordinator_receipt.json"
    _write_create_only(destination, _canonical_bytes(document) + b"\n")
    publication_guard()
    return destination


def member_work_order_map(
    work_orders: Sequence[Path],
) -> dict[tuple[str, int], Path]:
    """Return the exact arm/seed mapping after structural validation by name."""
    if len(work_orders) != CONTROL_MEMBER_COUNT:
        raise Stage09ParallelError("member work-order list is incomplete")
    result: dict[tuple[str, int], Path] = {}
    for member, path in zip(expected_control_members(), work_orders, strict=True):
        expected_suffix = Path(member.work_order_relative)
        if tuple(path.parts[-len(expected_suffix.parts) :]) != expected_suffix.parts:
            raise Stage09ParallelError("member work-order ordering changed")
        key = (member.arm_id, member.seed)
        if key in result:
            raise Stage09ParallelError("duplicate member work-order key")
        result[key] = path
    return result


__all__ = [
    "AUTHORIZATION_FORMAT",
    "CONTROL_MEMBER_COUNT",
    "CONTROL_PRECOMPUTE_DIRECTORY",
    "CONTROL_RESUME_DIRECTORY",
    "COORDINATOR_RECEIPT_FORMAT",
    "ControlMember",
    "DEFAULT_CONTROL_WORKERS",
    "MAX_CONTROL_WORKERS",
    "MEMBER_RECEIPT_FORMAT",
    "ModelMatrixGate",
    "RECOMMENDED_MAX_CONTROL_WORKERS",
    "SEMANTIC_VALIDATION_FORMAT",
    "Stage09ParallelError",
    "ValidatedWorkOrder",
    "WORK_ORDER_FORMAT",
    "execute_work_orders_bounded",
    "expected_control_members",
    "extracted_member_payload",
    "freeze_control_matrix_receipt",
    "freeze_control_plan",
    "materialize_member_resume_checkpoint",
    "member_work_order_map",
    "publish_control_member",
    "validate_live_work_order",
    "validate_member_archive",
    "validate_stage09_model_matrix_gate",
    "validate_work_order_structure",
]
