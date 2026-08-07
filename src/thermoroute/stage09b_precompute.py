"""Fail-closed member-level precomputation for the 45 Stage-09b members.

Parallelism is an execution detail only.  This module freezes the same nine-arm,
five-seed scientific matrix consumed by the serial Stage-09b entry point, gives
each member an exclusive deterministic namespace, and commits a member only
after checkpoint-to-prediction replay has succeeded.  A retry can reuse a
committed member only after the caller has repeated that exact replay.

The run-local scheduling protocol has three layers:

* one parent authorization plus 45 immutable member work orders;
* one create-only canonical receipt per completed member; and
* one create-only coordinator receipt after an independent prediction-semantic
  matrix replay.

These receipts coordinate resumable work; they are not final scientific
authority.  Only the separately published ``route_a_stage09b_completion.json``
receipt, whose verifier independently reconstructs windows and replays every
checkpoint best state, can authorize downstream scientific evidence.

Worker count, scheduling order, PIDs, timestamps, and hostnames are deliberately
absent from every scientific identity.  Source bytes, numerical runtime, fixed
development-input closure, model-matrix amendment/seal, arm configuration,
member seed, and every committed artifact byte are explicitly bound.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any

from .input_closure import InputClosure, resolve_development_input_closure
from .model_matrix_amendment import (
    AMENDMENT_FORMAT,
    AMENDMENT_ID,
    AMENDMENT_RELATIVE,
    AMENDMENT_SEAL_FORMAT,
    AMENDMENT_SEAL_RELATIVE,
    AMENDMENT_SEAL_STATUS,
    AMENDMENT_STATUS,
    PLAIN_CONTROL_ALLOWED_INPUTS,
    PLAIN_CONTROL_FORBIDDEN_INPUTS,
    SEEDS as MODEL_MATRIX_SEEDS,
    STAGE09B_ARMS,
    model_matrix_contract_id,
    validate_model_matrix_amendment,
    validate_model_matrix_amendment_seal,
)
from .repro import (
    RUN_SCHEMA_VERSION,
    RunIdentity,
    assert_formal_numerical_policy,
    numerical_policy_document_sha256,
    numerical_policy_role_cap,
    numerical_runtime_contract,
    sha256_file,
    sha256_json,
)


AUTHORIZATION_FORMAT = "thermoroute.stage09b-member-authorization.v1"
WORK_ORDER_FORMAT = "thermoroute.stage09b-member-work-order.v1"
MEMBER_RECEIPT_FORMAT = "thermoroute.stage09b-member-receipt.v1"
COORDINATOR_RECEIPT_FORMAT = "thermoroute.stage09b-member-coordinator.v1"
MATRIX_BINDING_FORMAT = "thermoroute.stage09b-model-matrix-binding.v1"
PRECOMPUTE_DIRECTORY = "stage09b_member_precompute_v1"
MEMBER_LOCK_DIRECTORY = "stage09b_member_execution_locks_v1"
EXPECTED_MEMBER_COUNT = 45
MAX_PARALLEL_WORKERS = 96
_MAX_JSON_BYTES = 4 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{20}")
_CREATE_ONLY_TEMP_TAG = "thermoroute-create-only"
_INTERMEDIATE_EVIDENCE_ROLE = (
    "INTERMEDIATE_SCHEDULING_ONLY_NOT_FINAL_SCIENTIFIC_AUTHORITY"
)

_PLAIN_ARMS = ("PlainMLP-7var", "PlainCausalTCN-7var")
_FULL_ARM = "ThermoRoute-ladder-07_plus_WDSP"
_EXPECTED_PARAMETER_COUNTS = {
    "PlainMLP-7var": 38_860,
    "PlainCausalTCN-7var": 38_346,
    _FULL_ARM: 38_505,
}
_FORMAL_CONFIG_KEYS = {
    "stage",
    "format",
    "execution_role",
    "evidence_role",
    "development_disclosure",
    "panel_date_range",
    "development_evaluation_interval",
    "blind_or_confirmatory",
    "suite_pointer_written",
    "training_device",
    "variables",
    "context_length",
    "horizons",
    "time_split",
    "station_sampling",
    "selection_metric",
    "train_config",
    "arms",
    "expected_member_registry",
    "parameter_counts",
    "architecture_templates",
    "parameter_match_tolerance_fraction",
    "architecture_candidates_per_arm",
    "historical_tuning_budget_equalized",
    "development_predictor_bridge",
    "formal_numerical_policy",
    "eval_batch_size",
    "input_closure_sha256",
    "input_closure_file_count",
}


class Stage09bPrecomputeError(RuntimeError):
    """A Stage-09b authorization, member, or exact matrix failed closed."""


@dataclass(frozen=True, order=True)
class Stage09bMember:
    """One arm-major, seed-minor member of the frozen 9 x 5 matrix."""

    arm_index: int
    arm_id: str
    seed: int

    @property
    def member_id(self) -> str:
        return f"arm{self.arm_index:02d}-seed{self.seed}"

    @property
    def work_order_relative(self) -> str:
        return f"work_orders/arm{self.arm_index:02d}/seed{self.seed}.json"

    @property
    def receipt_relative(self) -> str:
        return f"member_receipts/arm{self.arm_index:02d}/seed{self.seed}.json"

    @property
    def lock_relative(self) -> str:
        # Existing repository policy ignores ``.*.formal-run.lock`` under any
        # content-addressed run.  Keep mutable PID/time diagnostics out of Git
        # and the release while retaining one deterministic POSIX lock path.
        return f"arm{self.arm_index:02d}/.seed{self.seed}.formal-run.lock"

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "arm_index": self.arm_index,
            "arm_id": self.arm_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class Stage09bMatrixGate:
    """Exact outcome-free model-matrix objects validated by a parent."""

    amendment: Mapping[str, Any]
    seal: Mapping[str, Any]
    binding: Mapping[str, Any]
    contract_id: str

    def assert_bytes_unchanged(self, root: str | Path) -> None:
        repository = Path(root).resolve()
        _assert_file_binding(
            repository,
            self.binding.get("amendment"),
            label="model-matrix amendment",
        )
        _assert_file_binding(
            repository,
            self.binding.get("seal"),
            label="model-matrix amendment seal",
        )


@dataclass(frozen=True)
class ValidatedStage09bWorkOrder:
    """A live-gated work order safe to execute and publish once."""

    root: Path
    path: Path
    document: Mapping[str, Any]
    authorization_path: Path
    authorization: Mapping[str, Any]
    identity: RunIdentity
    input_closure: InputClosure
    matrix_gate: Stage09bMatrixGate

    @property
    def member(self) -> Stage09bMember:
        return _member_from_mapping(self.document.get("member"))

    @property
    def run_directory(self) -> Path:
        return _run_directory(self.root, self.identity.run_id)

    @property
    def receipt_path(self) -> Path:
        return _precompute_root(self.root, self.identity.run_id) / self.member.receipt_relative

    @property
    def artifact_paths(self) -> dict[str, Path]:
        value = self.document.get("artifacts")
        if not isinstance(value, Mapping):  # validated during construction
            raise Stage09bPrecomputeError("member artifact registry is malformed")
        return {
            key: _safe_relative_file(self.root, str(relative), require_exists=False)
            for key, relative in value.items()
        }

    def assert_unchanged(self) -> None:
        """Recheck all scientific identity inputs immediately before commit."""
        assert_formal_numerical_policy(require_hash_randomization=True)
        self.input_closure.assert_unchanged()
        if self.input_closure.binding_digest != self.identity.input_closure_sha256:
            raise Stage09bPrecomputeError("development-input closure changed")
        if sha256_json(numerical_runtime_contract()) != self.identity.runtime_sha256:
            raise Stage09bPrecomputeError("Stage-09b numerical runtime changed")
        self.matrix_gate.assert_bytes_unchanged(self.root)
        _assert_file_binding(
            self.root,
            self.document.get("authorization"),
            label="member authorization",
        )
        if _load_json(
            self.path, label="member work order", canonical=True
        ) != self.document:
            raise Stage09bPrecomputeError("member work-order bytes changed")
        _validate_run_manifest(
            self.root,
            self.identity,
            self.authorization.get("resolved_config"),
        )


def expected_stage09b_precompute_members() -> tuple[Stage09bMember, ...]:
    members = tuple(
        Stage09bMember(arm_index, arm_id, int(seed))
        for arm_index, arm_id in enumerate(STAGE09B_ARMS)
        for seed in MODEL_MATRIX_SEEDS
    )
    if len(members) != EXPECTED_MEMBER_COUNT:  # pragma: no cover - constants
        raise AssertionError("Stage-09b registry is not 9 x 5")
    return members


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage09bPrecomputeError("value is not canonical JSON") from exc


def _with_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise Stage09bPrecomputeError(f"self-hash field already exists: {field}")
    stable = dict(value)
    return {**stable, field: hashlib.sha256(_canonical_bytes(stable)).hexdigest()}


def _validate_self_hash(
    value: object,
    *,
    field: str,
    keys: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys | {field}:
        raise Stage09bPrecomputeError(f"{label} schema is not exact")
    digest = value.get(field)
    stable = {key: item for key, item in value.items() if key != field}
    if (
        type(digest) is not str
        or _DIGEST.fullmatch(digest) is None
        or digest != hashlib.sha256(_canonical_bytes(stable)).hexdigest()
    ):
        raise Stage09bPrecomputeError(f"{label} self-hash differs")
    return value


def _normal_relative(value: str, *, label: str) -> str:
    candidate = Path(value)
    if (
        not value
        or candidate.is_absolute()
        or "\\" in value
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise Stage09bPrecomputeError(f"{label} is not a canonical relative path")
    return value


def _relative(root: Path, path: Path, *, label: str) -> str:
    root = root.resolve()
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise Stage09bPrecomputeError(f"{label} escapes the repository") from exc
    return _normal_relative(relative, label=label)


def _safe_relative_file(
    root: Path,
    relative: str,
    *,
    require_exists: bool = True,
) -> Path:
    relative = _normal_relative(relative, label="file path")
    root = root.resolve()
    parts = Path(relative).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if require_exists or index < len(parts) - 1:
                raise Stage09bPrecomputeError(f"required path is absent: {relative}")
            return current
        if stat.S_ISLNK(info.st_mode):
            raise Stage09bPrecomputeError(f"path contains a symlink: {relative}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise Stage09bPrecomputeError(f"path crosses a non-directory: {relative}")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise Stage09bPrecomputeError(f"path is not a single-link regular file: {relative}")
    return root / relative


def _safe_directory(path: Path, *, parent: Path | None = None) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if parent is not None and lexical.parent != parent.resolve():
        raise Stage09bPrecomputeError("directory has an unexpected parent")
    if os.path.lexists(lexical):
        info = lexical.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise Stage09bPrecomputeError(f"unsafe directory collision: {lexical}")
    else:
        lexical.mkdir(mode=0o700)
    return lexical


def _mkdir_chain(root: Path, relative: str) -> Path:
    current = root.resolve()
    for part in Path(_normal_relative(relative, label="directory path")).parts:
        current = _safe_directory(current / part, parent=current)
    return current


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_temp_path(path: Path, nonce: str) -> Path:
    if _DIGEST.fullmatch(nonce) is None:
        raise Stage09bPrecomputeError("create-only temporary nonce is malformed")
    return path.with_name(
        f".{path.name}.{_CREATE_ONLY_TEMP_TAG}.{nonce}.tmp"
    )


def _create_only_temp_candidates(path: Path) -> list[Path]:
    prefix = f".{path.name}.{_CREATE_ONLY_TEMP_TAG}."
    suffix = ".tmp"
    candidates: list[Path] = []
    for candidate in path.parent.iterdir():
        name = candidate.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        nonce = name[len(prefix) : -len(suffix)]
        if _DIGEST.fullmatch(nonce) is not None:
            candidates.append(candidate)
    return sorted(candidates)


def _read_owner_private_temp(
    path: Path,
    *,
    parent_device: int,
    maximum_bytes: int,
) -> tuple[os.stat_result, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"cannot inspect create-only temporary: {path}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_dev != parent_device
        ):
            raise Stage09bPrecomputeError(
                f"create-only temporary is not owner-private: {path}"
            )
        if info.st_size > maximum_bytes:
            raise Stage09bPrecomputeError(
                f"create-only temporary is unexpectedly large: {path}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only temporary changed during inspection: {path}"
        ) from exc
    if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
        raise Stage09bPrecomputeError(
            f"create-only temporary identity changed: {path}"
        )
    return info, b"".join(chunks)


def _inspect_owner_private_temp(
    path: Path,
    *,
    parent_device: int,
) -> os.stat_result:
    """Inspect an orphan without trusting its possibly partial payload bytes."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"cannot inspect create-only temporary: {path}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_dev != parent_device
        ):
            raise Stage09bPrecomputeError(
                f"create-only temporary is not owner-private: {path}"
            )
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only temporary changed during inspection: {path}"
        ) from exc
    if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
        raise Stage09bPrecomputeError(
            f"create-only temporary identity changed: {path}"
        )
    return current


def _unlink_verified_temp(
    path: Path,
    *,
    expected: os.stat_result,
) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only temporary disappeared before cleanup: {path}"
        ) from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino)
        or current.st_uid != os.getuid()
        or stat.S_IMODE(current.st_mode) & 0o077
        or current.st_nlink != expected.st_nlink
    ):
        raise Stage09bPrecomputeError(
            f"create-only temporary changed before cleanup: {path}"
        )
    path.unlink()


def _exact_final_bytes(path: Path, payload: bytes) -> os.stat_result:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only final is absent after publication: {path}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise Stage09bPrecomputeError(f"unsafe create-only collision: {path}")
        if info.st_size != len(payload):
            raise Stage09bPrecomputeError(f"create-only byte collision: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"cannot inspect create-only final: {path}"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only final changed during inspection: {path}"
        ) from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
        or current.st_nlink != after_read.st_nlink
        or current.st_size != after_read.st_size
    ):
        raise Stage09bPrecomputeError(
            f"create-only final changed during inspection: {path}"
        )
    if b"".join(chunks) != payload:
        raise Stage09bPrecomputeError(f"create-only byte collision: {path}")
    return current


def _recover_create_only_state(path: Path, payload: bytes) -> bool:
    """Recover only internally identifiable hard-link publication crash windows."""
    parent_info = path.parent.lstat()
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise Stage09bPrecomputeError("create-only parent directory is unsafe")
    candidates = _create_only_temp_candidates(path)
    final_exists = os.path.lexists(path)
    if not final_exists:
        orphans = [
            (
                candidate,
                _inspect_owner_private_temp(
                    candidate, parent_device=parent_info.st_dev
                ),
            )
            for candidate in candidates
        ]
        for candidate, info in orphans:
            if info.st_nlink != 1:
                raise Stage09bPrecomputeError(
                    "create-only orphan temporary has an external hard link"
                )
            _unlink_verified_temp(candidate, expected=info)
        if orphans:
            _fsync_directory(path.parent)
        if os.path.lexists(path):
            raise Stage09bPrecomputeError(
                f"concurrent create-only collision: {path}"
            )
        return False

    inspected = [
        (
            candidate,
            *_read_owner_private_temp(
                candidate,
                parent_device=parent_info.st_dev,
                maximum_bytes=len(payload),
            ),
        )
        for candidate in candidates
    ]
    final = _exact_final_bytes(path, payload)
    if final.st_nlink > 2:
        raise Stage09bPrecomputeError(
            "create-only final has an external or excessive hard link"
        )
    matching = [
        (candidate, info, content)
        for candidate, info, content in inspected
        if (info.st_dev, info.st_ino) == (final.st_dev, final.st_ino)
    ]
    if final.st_nlink == 2:
        if (
            len(inspected) != 1
            or len(matching) != 1
            or matching[0][1].st_nlink != 2
            or matching[0][2] != payload
        ):
            raise Stage09bPrecomputeError(
                "hard-linked create-only final lacks one exact internal stale temporary"
            )
        _unlink_verified_temp(matching[0][0], expected=matching[0][1])
        _fsync_directory(path.parent)
        recovered = _exact_final_bytes(path, payload)
        if recovered.st_nlink != 1:
            raise Stage09bPrecomputeError(
                "create-only final did not recover to one link"
            )
        return True
    if final.st_nlink != 1:
        raise Stage09bPrecomputeError("create-only final link count is invalid")
    for candidate, info, content in inspected:
        if info.st_nlink != 1 or content != payload:
            raise Stage09bPrecomputeError(
                "create-only stale temporary is unsafe"
            )
        _unlink_verified_temp(candidate, expected=info)
    if inspected:
        _fsync_directory(path.parent)
    stable = _exact_final_bytes(path, payload)
    if stable.st_nlink != 1:
        raise Stage09bPrecomputeError("create-only final link count changed")
    return True


def _validated_pending_create_only_temps(path: Path) -> set[Path]:
    """Classify only safe, unfinished internal publication names.

    This is a namespace-audit helper, not a recovery shortcut.  It never
    deletes or accepts final bytes.  The destination's later
    :func:`_publish_create_only` call must still validate the expected payload
    before it removes a linked temporary or reuses the final file.
    """
    parent_info = path.parent.lstat()
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise Stage09bPrecomputeError("create-only parent directory is unsafe")
    try:
        final = path.lstat()
    except FileNotFoundError:
        final = None
    pending: set[Path] = set()
    for candidate in _create_only_temp_candidates(path):
        info = _inspect_owner_private_temp(
            candidate,
            parent_device=parent_info.st_dev,
        )
        if info.st_nlink == 1:
            pending.add(candidate)
            continue
        if (
            info.st_nlink == 2
            and final is not None
            and not stat.S_ISLNK(final.st_mode)
            and stat.S_ISREG(final.st_mode)
            and final.st_nlink == 2
            and (info.st_dev, info.st_ino) == (final.st_dev, final.st_ino)
        ):
            pending.add(candidate)
            continue
        raise Stage09bPrecomputeError(
            "create-only pending temporary has an external or invalid hard link"
        )
    return pending


def _publish_create_only(
    path: Path,
    payload: bytes,
    *,
    _fault_injector: Callable[[str, Path, Path], object] | None = None,
) -> None:
    """Publish once, recovering only exact internal crash state.

    The caller must hold the enclosing run/parent/member transaction lock.  That
    lock makes exact-name orphan cleanup safe: another live publisher for this
    destination cannot own one of the matching temporary paths concurrently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _recover_create_only_state(path, payload):
        return
    temporary: Path | None = None
    descriptor: int | None = None
    for _attempt in range(32):
        candidate = _create_only_temp_path(path, secrets.token_hex(32))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        temporary = candidate
        break
    if temporary is None or descriptor is None:  # pragma: no cover - nonce exhaustion
        raise Stage09bPrecomputeError("cannot allocate create-only temporary")
    preserve_crash_state = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Stage09bPrecomputeError(
                f"concurrent create-only collision: {path}"
            ) from exc
        if _fault_injector is not None:
            try:
                _fault_injector("after_link_before_temp_unlink", temporary, path)
            except BaseException:
                # Simulate process death: a real SIGKILL would bypass ``finally``.
                preserve_crash_state = True
                raise
        temporary_info = temporary.lstat()
        _unlink_verified_temp(temporary, expected=temporary_info)
        temporary = None
        _fsync_directory(path.parent)
        final = _exact_final_bytes(path, payload)
        if final.st_nlink != 1:
            raise Stage09bPrecomputeError(
                "create-only publication retained an unexpected hard link"
            )
    finally:
        if temporary is not None and not preserve_crash_state:
            info = _inspect_owner_private_temp(
                temporary,
                parent_device=path.parent.lstat().st_dev,
            )
            _unlink_verified_temp(temporary, expected=info)
            _fsync_directory(path.parent)


def _artifact_final_state(
    path: Path,
    *,
    parent_device: int,
) -> os.stat_result:
    """Open one create-only artifact without following its final component."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"cannot inspect create-only artifact: {path}"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_dev != parent_device
        ):
            raise Stage09bPrecomputeError(
                f"create-only artifact is foreign or non-regular: {path}"
            )
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only artifact changed during inspection: {path}"
        ) from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
        or current.st_nlink != info.st_nlink
        or current.st_size != info.st_size
    ):
        raise Stage09bPrecomputeError(
            f"create-only artifact changed during inspection: {path}"
        )
    return current


def _artifact_fingerprint(
    path: Path,
    *,
    parent_device: int,
    expected_links: int,
    require_owner_private: bool,
) -> tuple[os.stat_result, str]:
    """Hash a potentially large artifact through one verified no-follow fd."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"cannot fingerprint create-only artifact: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_dev != parent_device
            or before.st_nlink != expected_links
            or (require_owner_private and stat.S_IMODE(before.st_mode) & 0o077)
        ):
            raise Stage09bPrecomputeError(
                f"create-only artifact fingerprint input is unsafe: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            f"create-only artifact changed during fingerprint: {path}"
        ) from exc
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_nlink) != identity
        or (current.st_dev, current.st_ino, current.st_size, current.st_nlink)
        != identity
        or not stat.S_ISREG(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
    ):
        raise Stage09bPrecomputeError(
            f"create-only artifact changed during fingerprint: {path}"
        )
    return current, digest.hexdigest()


def recover_create_only_artifact_state(path: str | Path) -> bool:
    """Recover one large-file create-only publication crash window.

    The caller must hold the member or parent transaction lock.  Only an exact
    owner-private temporary name created beside this destination can explain a
    two-link final.  External links, symlinks, foreign files, and ambiguous
    temporary sets fail closed.  The artifact's scientific bytes are validated
    by its normal sidecar/replay path after this filesystem-only recovery.
    """
    destination = Path(path)
    try:
        parent = destination.parent.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            "create-only artifact parent directory is absent"
        ) from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise Stage09bPrecomputeError(
            "create-only artifact parent directory is unsafe"
        )
    candidates = _create_only_temp_candidates(destination)
    if not os.path.lexists(destination):
        inspected = [
            (
                candidate,
                _inspect_owner_private_temp(
                    candidate,
                    parent_device=parent.st_dev,
                ),
            )
            for candidate in candidates
        ]
        for candidate, info in inspected:
            if info.st_nlink != 1:
                raise Stage09bPrecomputeError(
                    "orphan create-only artifact temporary has an external link"
                )
            _unlink_verified_temp(candidate, expected=info)
        if inspected:
            _fsync_directory(destination.parent)
        if os.path.lexists(destination):
            raise Stage09bPrecomputeError(
                f"concurrent create-only artifact collision: {destination}"
            )
        return False

    final = _artifact_final_state(destination, parent_device=parent.st_dev)
    inspected = [
        (
            candidate,
            _inspect_owner_private_temp(
                candidate,
                parent_device=parent.st_dev,
            ),
        )
        for candidate in candidates
    ]
    matching = [
        (candidate, info)
        for candidate, info in inspected
        if (info.st_dev, info.st_ino) == (final.st_dev, final.st_ino)
    ]
    if final.st_nlink == 2:
        if (
            len(inspected) != 1
            or len(matching) != 1
            or matching[0][1].st_nlink != 2
        ):
            raise Stage09bPrecomputeError(
                "hard-linked create-only artifact lacks one exact internal temporary"
            )
        _unlink_verified_temp(matching[0][0], expected=matching[0][1])
        _fsync_directory(destination.parent)
    elif final.st_nlink == 1:
        for candidate, info in inspected:
            if info.st_nlink != 1:
                raise Stage09bPrecomputeError(
                    "stale create-only artifact temporary has an external link"
                )
            _unlink_verified_temp(candidate, expected=info)
        if inspected:
            _fsync_directory(destination.parent)
    else:
        raise Stage09bPrecomputeError(
            "create-only artifact has an external or excessive hard link"
        )
    stable = _artifact_final_state(destination, parent_device=parent.st_dev)
    if stable.st_nlink != 1:
        raise Stage09bPrecomputeError(
            "create-only artifact did not recover to one link"
        )
    return True


def allocate_create_only_artifact_temp(path: str | Path) -> tuple[int, Path]:
    """Allocate one exact owner-private staging name beside an absent final."""
    destination = Path(path)
    try:
        parent = destination.parent.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            "create-only artifact parent directory is absent"
        ) from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise Stage09bPrecomputeError(
            "create-only artifact parent directory is unsafe"
        )
    if os.path.lexists(destination) or _create_only_temp_candidates(destination):
        raise Stage09bPrecomputeError(
            "create-only artifact state must be recovered before staging"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _attempt in range(32):
        temporary = _create_only_temp_path(destination, secrets.token_hex(32))
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_dev != parent.st_dev
            or info.st_nlink != 1
        ):
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise Stage09bPrecomputeError(
                "allocated create-only artifact temporary is unsafe"
            )
        return descriptor, temporary
    raise Stage09bPrecomputeError(
        "cannot allocate create-only artifact temporary"
    )


def publish_create_only_artifact_from_temp(
    temporary: str | Path,
    destination: str | Path,
    *,
    publication_guard: Callable[[], object],
    _fault_injector: Callable[[str, Path, Path], object] | None = None,
) -> None:
    """Create-only publish a staged large file and verify its streamed digest."""
    temporary_path, destination_path = Path(temporary), Path(destination)
    try:
        parent = destination_path.parent.lstat()
    except OSError as exc:
        raise Stage09bPrecomputeError(
            "create-only artifact parent directory is absent"
        ) from exc
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or temporary_path.parent != destination_path.parent
        or temporary_path not in _create_only_temp_candidates(destination_path)
        or len(_create_only_temp_candidates(destination_path)) != 1
        or os.path.lexists(destination_path)
    ):
        raise Stage09bPrecomputeError(
            "create-only artifact staging namespace is unsafe"
        )
    staged, staged_sha256 = _artifact_fingerprint(
        temporary_path,
        parent_device=parent.st_dev,
        expected_links=1,
        require_owner_private=True,
    )
    publication_guard()
    try:
        os.link(temporary_path, destination_path)
    except FileExistsError as exc:
        raise Stage09bPrecomputeError(
            f"concurrent create-only artifact collision: {destination_path}"
        ) from exc
    if _fault_injector is not None:
        _fault_injector(
            "after_link_before_temp_unlink",
            temporary_path,
            destination_path,
        )
    linked = temporary_path.lstat()
    if (
        (linked.st_dev, linked.st_ino) != (staged.st_dev, staged.st_ino)
        or linked.st_nlink != 2
    ):
        raise Stage09bPrecomputeError(
            "create-only artifact temporary changed after link"
        )
    _unlink_verified_temp(temporary_path, expected=linked)
    _fsync_directory(destination_path.parent)
    final, final_sha256 = _artifact_fingerprint(
        destination_path,
        parent_device=parent.st_dev,
        expected_links=1,
        require_owner_private=False,
    )
    if final.st_size != staged.st_size or final_sha256 != staged_sha256:
        raise Stage09bPrecomputeError(
            "create-only artifact differs from its staged bytes"
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Stage09bPrecomputeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> Any:
    raise Stage09bPrecomputeError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, *, label: str, canonical: bool = False) -> Any:
    try:
        info = path.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > _MAX_JSON_BYTES
        ):
            raise Stage09bPrecomputeError(f"{label} is not a bounded regular file")
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage09bPrecomputeError(f"{label} is not valid JSON") from exc
    if canonical and raw != _canonical_bytes(value) + b"\n":
        raise Stage09bPrecomputeError(f"{label} bytes are not canonical")
    return value


def _file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_relative_file(root, relative)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _assert_file_binding(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "bytes"}:
        raise Stage09bPrecomputeError(f"{label} binding schema changed")
    relative, digest, byte_count = value.get("path"), value.get("sha256"), value.get("bytes")
    if (
        type(relative) is not str
        or type(digest) is not str
        or _DIGEST.fullmatch(digest) is None
        or type(byte_count) is not int
        or byte_count < 0
    ):
        raise Stage09bPrecomputeError(f"{label} binding is malformed")
    path = _safe_relative_file(root, relative)
    if path.stat().st_size != byte_count or sha256_file(path) != digest:
        raise Stage09bPrecomputeError(f"{label} bytes changed")
    return path


def _require_safe_control_file(root: Path, path: Path, *, label: str) -> Path:
    relative = _relative(root, path, label=label)
    safe = _safe_relative_file(root, relative)
    if safe != path:
        raise Stage09bPrecomputeError(f"{label} path is noncanonical")
    return safe


def _run_directory(root: Path, run_id: str) -> Path:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise Stage09bPrecomputeError("Stage-09b run_id is not canonical")
    return root.resolve() / "outputs" / "runs" / "09b_development_controls" / run_id


def _precompute_root(root: Path, run_id: str) -> Path:
    return _run_directory(root, run_id) / PRECOMPUTE_DIRECTORY


def _member_from_mapping(value: object) -> Stage09bMember:
    if not isinstance(value, Mapping) or set(value) != {
        "member_id", "arm_index", "arm_id", "seed"
    }:
        raise Stage09bPrecomputeError("member identity schema changed")
    if type(value.get("arm_index")) is not int or type(value.get("seed")) is not int:
        raise Stage09bPrecomputeError("member identity is malformed")
    member = Stage09bMember(
        int(value["arm_index"]), str(value.get("arm_id")), int(value["seed"])
    )
    if member not in expected_stage09b_precompute_members() or value != member.as_dict():
        raise Stage09bPrecomputeError("member is outside the frozen 45-member matrix")
    return member


def _identity_from_mapping(value: object) -> RunIdentity:
    fields = {
        "run_id", "panel_sha256", "registry_sha256", "config_sha256",
        "source_sha256", "runtime_sha256", "input_closure_sha256",
        "schema_version",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Stage09bPrecomputeError("run identity schema changed")
    try:
        identity = RunIdentity(**{key: str(item) for key, item in value.items()})
    except (TypeError, ValueError) as exc:
        raise Stage09bPrecomputeError("run identity is malformed") from exc
    if identity.schema_version != RUN_SCHEMA_VERSION or _RUN_ID.fullmatch(identity.run_id) is None:
        raise Stage09bPrecomputeError("run identity version/id changed")
    for field in (
        "panel_sha256", "registry_sha256", "config_sha256", "source_sha256",
        "runtime_sha256", "input_closure_sha256",
    ):
        if _DIGEST.fullmatch(getattr(identity, field)) is None:
            raise Stage09bPrecomputeError("run identity digest is malformed")
    parts = {
        "schema_version": identity.schema_version,
        "panel_sha256": identity.panel_sha256,
        "registry_sha256": identity.registry_sha256,
        "config_sha256": identity.config_sha256,
        "source_sha256": identity.source_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "input_closure_sha256": identity.input_closure_sha256,
    }
    if identity.run_id != sha256_json(parts)[:20]:
        raise Stage09bPrecomputeError("run identity is not content-addressed")
    return identity


def _expected_comparisons() -> list[dict[str, Any]]:
    full = _FULL_ARM
    pairs = [
        ("full_vs_control", f"{full}-minus-{reference}", full, reference)
        for reference in _PLAIN_ARMS
    ]
    pairs.extend(
        ("adjacent_feature_ladder", f"{candidate}-minus-{reference}", candidate, reference)
        for reference, candidate in zip(STAGE09B_ARMS[2:-1], STAGE09B_ARMS[3:], strict=True)
    )
    return [
        {
            "comparison_family": family,
            "comparison_id": comparison_id,
            "candidate_arm_id": candidate,
            "reference_arm_id": reference,
            "seeds": list(MODEL_MATRIX_SEEDS),
        }
        for family, comparison_id, candidate, reference in pairs
    ]


def _validate_matrix_document(amendment: Mapping[str, Any]) -> Mapping[str, Any]:
    matrix = amendment.get("stage09b_development_control_matrix")
    if not isinstance(matrix, Mapping):
        raise Stage09bPrecomputeError("model-matrix amendment lacks Stage-09b")
    fairness = matrix.get("plain_control_information_fairness_revision")
    boundary = matrix.get("interpretation_boundary")
    if (
        matrix.get("stage") != "09b"
        or matrix.get("role") != "DEVELOPMENT_ONLY_EXPLORATORY_NOT_BLIND_NOT_CONFIRMATORY"
        or matrix.get("development_interval") != ["2006-01-01", "2020-12-31"]
        or matrix.get("arms") != list(STAGE09B_ARMS)
        or matrix.get("arm_count") != 9
        or matrix.get("seeds_per_arm") != list(MODEL_MATRIX_SEEDS)
        or matrix.get("seed_count_per_arm") != 5
        or matrix.get("expected_member_count") != EXPECTED_MEMBER_COUNT
        or matrix.get("splits") != ["val", "calib", "test"]
        or matrix.get("horizons") != [1, 3, 7]
        or matrix.get("member_summary_cell_count") != 405
        or matrix.get("paired_comparisons") != _expected_comparisons()
        or matrix.get("paired_comparison_count") != 8
        or matrix.get("paired_seed_count") != 40
        or matrix.get("paired_effect_cell_count") != 360
        or not isinstance(fairness, Mapping)
        or fairness.get("affected_arms") != list(_PLAIN_ARMS)
        or fairness.get("change_timing") != "PROSPECTIVE_PRELABEL_BEFORE_RETRAINING"
        or fairness.get("allowed_consumed_batch_keys") != list(PLAIN_CONTROL_ALLOWED_INPUTS)
        or fairness.get("all_unlisted_batch_keys_forbidden_to_consume") is not True
        or fairness.get("explicitly_forbidden_batch_keys")
        != list(PLAIN_CONTROL_FORBIDDEN_INPUTS)
        or fairness.get("context_semantics")
        != "same_outcome_free_issue_time_and_derived_context_as_full_ThermoRoute"
        or fairness.get("common_anchor") != "damped_prior"
        or fairness.get("prediction_head")
        != "unrestricted_residual_added_to_damped_prior"
        or any(
            fairness.get(field) is not False
            for field in (
                "router_present", "mixture_of_experts_present",
                "physics_modules_present", "bounded_residual_present",
                "old_stage09b_artifacts_eligible_for_final_freeze",
                "outcome_or_result_based_retuning_allowed",
            )
        )
        or fairness.get("full_retraining_and_replay_required") is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("historical_tuning_budget_equalized") is not False
        or boundary.get("feature_ladder_order")
        != ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"]
        or boundary.get("adjacent_ladder_effect_is_path_dependent") is not True
        or any(
            boundary.get(field) is not False
            for field in (
                "independent_feature_importance_claim_allowed",
                "causal_attribution_allowed", "confirmatory_inference_allowed",
            )
        )
    ):
        raise Stage09bPrecomputeError("Stage-09b model-matrix contract changed")
    return matrix


def _validate_matrix_binding(value: object) -> Mapping[str, Any]:
    keys = {
        "format", "amendment", "seal", "amendment_format",
        "amendment_status", "seal_format", "seal_status", "amendment_id",
        "amendment_document_commit", "model_matrix_contract_id",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise Stage09bPrecomputeError("authorization model-matrix binding schema changed")
    if (
        value.get("format") != MATRIX_BINDING_FORMAT
        or value.get("amendment_format") != AMENDMENT_FORMAT
        or value.get("amendment_status") != AMENDMENT_STATUS
        or value.get("seal_format") != AMENDMENT_SEAL_FORMAT
        or value.get("seal_status") != AMENDMENT_SEAL_STATUS
        or value.get("amendment_id") != AMENDMENT_ID
        or type(value.get("amendment_document_commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", value["amendment_document_commit"]) is None
        or type(value.get("model_matrix_contract_id")) is not str
        or _DIGEST.fullmatch(value["model_matrix_contract_id"]) is None
    ):
        raise Stage09bPrecomputeError("authorization model-matrix binding changed")
    for label in ("amendment", "seal"):
        binding = value.get(label)
        if not isinstance(binding, Mapping) or set(binding) != {
            "path", "sha256", "bytes"
        }:
            raise Stage09bPrecomputeError(
                f"authorization model-matrix {label} binding changed"
            )
    return value


def validate_stage09b_model_matrix_gate(root: str | Path) -> Stage09bMatrixGate:
    """Replay the outcome-free amendment, future seal, bytes, and Git lineage."""
    repository = Path(root).resolve()
    try:
        amendment = validate_model_matrix_amendment(
            repository / AMENDMENT_RELATIVE, root=repository
        )
        seal = validate_model_matrix_amendment_seal(
            repository / AMENDMENT_SEAL_RELATIVE,
            root=repository,
            amendment_path=repository / AMENDMENT_RELATIVE,
        )
    except Exception as exc:
        raise Stage09bPrecomputeError("Stage-09b model-matrix gate failed") from exc
    matrix = _validate_matrix_document(amendment)
    stage09 = amendment.get("stage09_architecture_control_matrix")
    if not isinstance(stage09, Mapping):
        raise Stage09bPrecomputeError("model-matrix amendment lacks Stage-09")
    contract_id = model_matrix_contract_id(stage09, matrix)
    binding = {
        "format": MATRIX_BINDING_FORMAT,
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
    return Stage09bMatrixGate(amendment, seal, binding, contract_id)


def _validate_plain_template(template: object, *, arm_id: str) -> None:
    if not isinstance(template, Mapping):
        raise Stage09bPrecomputeError(f"{arm_id} architecture template is malformed")
    constructor = template.get("constructor_kwargs")
    if (
        template.get("information_contract") != "stage09b_outcome_free_issue_time_v1"
        or template.get("input_keys_read") != list(PLAIN_CONTROL_ALLOWED_INPUTS)
        or template.get("input_keys_explicitly_excluded")
        != list(PLAIN_CONTROL_FORBIDDEN_INPUTS)
        or template.get("future_keys_never_read") != list(PLAIN_CONTROL_FORBIDDEN_INPUTS)
        or template.get("common_anchor") != "damped_prior"
        or template.get("forecast_parameterization")
        != "damped_prior_plus_unrestricted_additive_neural_residual"
        or template.get("uses_dynamic_lag_router") is not False
        or template.get("uses_mixture_of_experts") is not False
        or template.get("uses_learned_physics_prior") is not False
        or template.get("uses_bounded_residual") is not False
        or not isinstance(constructor, Mapping)
        or constructor.get("n_vars") != 7
        or constructor.get("context_length") != 32
        or constructor.get("horizons") != [1, 3, 7]
        or constructor.get("use_information_matched_context") is not True
        or constructor.get("init_seed") != "member_seed"
        or template.get("initialization_seed_policy")
        != "exact declared member seed"
    ):
        raise Stage09bPrecomputeError(f"{arm_id} information-fairness contract changed")


def _validate_formal_config(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FORMAL_CONFIG_KEYS:
        raise Stage09bPrecomputeError("Stage-09b resolved config schema changed")
    arms = value.get("arms")
    members = value.get("expected_member_registry")
    counts = value.get("parameter_counts")
    templates = value.get("architecture_templates")
    if not isinstance(arms, list) or len(arms) != len(STAGE09B_ARMS):
        raise Stage09bPrecomputeError("Stage-09b arm config is malformed")
    observed_arms: list[str] = []
    for arm, expected_id in zip(arms, STAGE09B_ARMS, strict=True):
        if (
            not isinstance(arm, Mapping)
            or set(arm) != {"arm_id", "family", "feature_set", "variables", "seeds"}
            or arm.get("arm_id") != expected_id
            or arm.get("seeds") != list(MODEL_MATRIX_SEEDS)
            or not isinstance(arm.get("variables"), list)
            or not arm["variables"]
        ):
            raise Stage09bPrecomputeError("Stage-09b arm/seed contract changed")
        observed_arms.append(expected_id)
    expected_members = [
        [member.arm_id, member.seed] for member in expected_stage09b_precompute_members()
    ]
    if (
        observed_arms != list(STAGE09B_ARMS)
        or members != expected_members
        or value.get("stage") != "09b_development_controls"
        or value.get("execution_role")
        != "prelabel_relative_to_unopened_post_2020_confirmation"
        or value.get("evidence_role") != "development_only_exploratory"
        or value.get("panel_date_range") != ["2006-01-01", "2020-12-31"]
        or value.get("development_evaluation_interval") != ["2019-01-01", "2020-12-31"]
        or value.get("blind_or_confirmatory") is not False
        or value.get("suite_pointer_written") is not False
        or value.get("training_device") != "cpu"
        or value.get("variables")
        != ["WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP"]
        or value.get("context_length") != 32
        or value.get("horizons") != [1, 3, 7]
        or value.get("station_sampling") != "balanced"
        or value.get("selection_metric") != "station_macro"
        or value.get("parameter_match_tolerance_fraction") != 0.02
        or value.get("architecture_candidates_per_arm") != 1
        or value.get("historical_tuning_budget_equalized") is not False
        or type(value.get("eval_batch_size")) is not int
        or int(value["eval_batch_size"]) < 1
        or not isinstance(value.get("train_config"), Mapping)
        or type(value["train_config"].get("max_epochs")) is not int
        or int(value["train_config"]["max_epochs"]) < 1
        or type(value["train_config"].get("patience")) is not int
        or int(value["train_config"]["patience"]) < 0
        or not isinstance(counts, Mapping)
        or not isinstance(templates, Mapping)
        or set(counts) != set(STAGE09B_ARMS)
        or set(templates) != set(STAGE09B_ARMS)
    ):
        raise Stage09bPrecomputeError("Stage-09b formal scientific config changed")
    for arm_id, expected in _EXPECTED_PARAMETER_COUNTS.items():
        if counts.get(arm_id) != expected:
            raise Stage09bPrecomputeError(f"{arm_id} parameter budget changed")
    reference = int(counts[_FULL_ARM])
    for arm_id in _PLAIN_ARMS:
        if abs(int(counts[arm_id]) - reference) / reference > 0.02:
            raise Stage09bPrecomputeError(f"{arm_id} exceeds the 2% parameter budget")
        _validate_plain_template(templates[arm_id], arm_id=arm_id)
    return value


def _scientific_member_config(
    config: Mapping[str, Any], member: Stage09bMember
) -> dict[str, Any]:
    arms = config["arms"]
    assert isinstance(arms, list)
    arm = arms[member.arm_index]
    counts = config["parameter_counts"]
    assert isinstance(counts, Mapping)
    return {
        **dict(config),
        "arm": dict(arm),
        "seed": member.seed,
        "trainable_parameters": int(counts[member.arm_id]),
    }


def _artifact_relatives(identity: RunIdentity, member: Stage09bMember) -> dict[str, str]:
    base = f"outputs/runs/09b_development_controls/{identity.run_id}"
    prediction = f"{base}/arm_predictions/{member.arm_id}/seed{member.seed}.parquet"
    checkpoint = f"{base}/checkpoints/{member.arm_id}/seed{member.seed}.pt"
    return {
        "prediction": prediction,
        "prediction_sidecar": prediction + ".meta.json",
        "checkpoint": checkpoint,
        "checkpoint_sidecar": checkpoint + ".meta.json",
    }


def _member_lock_relative(identity: RunIdentity, member: Stage09bMember) -> str:
    return (
        f"outputs/runs/09b_development_controls/{identity.run_id}/"
        f"{MEMBER_LOCK_DIRECTORY}/{member.lock_relative}"
    )


def _authorization_document(
    *,
    root: Path,
    identity: RunIdentity,
    config: Mapping[str, Any],
    gate: Stage09bMatrixGate,
) -> dict[str, Any]:
    matrix = [
        {
            **member.as_dict(),
            "work_order": member.work_order_relative,
            "receipt": member.receipt_relative,
            "execution_lock": _member_lock_relative(identity, member),
            "scientific_member_config_sha256": sha256_json(
                _scientific_member_config(config, member)
            ),
        }
        for member in expected_stage09b_precompute_members()
    ]
    stable = {
        "format": AUTHORIZATION_FORMAT,
        "status": "FROZEN_BEFORE_STAGE09B_MEMBER_EXECUTION",
        "evidence_role": _INTERMEDIATE_EVIDENCE_ROLE,
        "final_scientific_authority": False,
        "required_final_authority": "outputs/models/route_a_stage09b_completion.json",
        "stage": "09b_development_controls",
        "run_identity": identity.as_dict(),
        "resolved_config": dict(config),
        "model_matrix": dict(gate.binding),
        "matrix": matrix,
        "matrix_audit": {
            "arm_count": 9,
            "seed_count_per_arm": 5,
            "expected_member_count": EXPECTED_MEMBER_COUNT,
            "arm_major_seed_minor_order": True,
            "missing_duplicate_or_extra_member_allowed": False,
            "same_forecast_keys_and_y_true_required": True,
            "same_seed_pairing_required": True,
            "plain_control_information_fairness_required": True,
        },
        "scientific_execution_contract": {
            "member_process_thread_cap": numerical_policy_role_cap(root, "stage09b"),
            "numerical_policy_document_sha256": numerical_policy_document_sha256(root),
            "fixed_arm_seed_config_and_epoch_policy": True,
            "same_training_and_prediction_functions_as_serial_stage09b": True,
            "exclusive_deterministic_member_namespaces": True,
            "create_only_member_commit": True,
            "checkpoint_exact_replay_required_before_reuse": True,
            "worker_count_is_execution_only": True,
            "worker_count_enters_run_identity": False,
            "scheduling_order_enters_scientific_configuration": False,
        },
    }
    return _with_self_hash(stable, "authorization_self_sha256")


def _work_order_document(
    *, root: Path, authorization_path: Path, authorization: Mapping[str, Any],
    identity: RunIdentity, member: Stage09bMember,
) -> dict[str, Any]:
    config = authorization["resolved_config"]
    assert isinstance(config, Mapping)
    scientific = _scientific_member_config(config, member)
    stable = {
        "format": WORK_ORDER_FORMAT,
        "status": "AUTHORIZED_NOT_YET_COMMITTED",
        "evidence_role": _INTERMEDIATE_EVIDENCE_ROLE,
        "final_scientific_authority": False,
        "required_final_authority": "outputs/models/route_a_stage09b_completion.json",
        "stage": "09b_development_controls",
        "authorization": _file_binding(
            root, _relative(root, authorization_path, label="authorization path")
        ),
        "run_identity": identity.as_dict(),
        "model_matrix_contract_id": authorization["model_matrix"][
            "model_matrix_contract_id"
        ],
        "member": member.as_dict(),
        "scientific_member_config": scientific,
        "scientific_member_config_sha256": sha256_json(scientific),
        "artifacts": _artifact_relatives(identity, member),
        "execution_lock": _member_lock_relative(identity, member),
        "receipt": member.receipt_relative,
        "publication_contract": {
            "member_receipt_is_commit_point": True,
            "receipt_publication": "same_filesystem_create_only_hard_link",
            "prediction_and_checkpoint_exact_replay_required": True,
            "partial_or_colliding_member_fails_closed": True,
        },
    }
    return _with_self_hash(stable, "work_order_self_sha256")


def freeze_stage09b_precompute_plan(
    *,
    root: str | Path,
    run_directory: str | Path,
    identity: RunIdentity,
    resolved_config: Mapping[str, Any],
    matrix_gate: Stage09bMatrixGate,
    publication_guard: Callable[[], object],
) -> tuple[Path, tuple[Path, ...]]:
    """Create or byte-verify one authorization and the exact 45 work orders."""
    repository = Path(root).resolve()
    expected_run = _run_directory(repository, identity.run_id)
    if Path(os.path.abspath(os.fspath(run_directory))) != expected_run:
        raise Stage09bPrecomputeError("precompute run directory is noncanonical")
    config = _validate_formal_config(resolved_config)
    if identity.config_sha256 != sha256_json(config):
        raise Stage09bPrecomputeError("resolved config differs from RunIdentity")
    _validate_matrix_document(matrix_gate.amendment)
    _validate_run_manifest(repository, identity, config)
    publication_guard()
    precompute = _safe_directory(expected_run / PRECOMPUTE_DIRECTORY, parent=expected_run)
    _mkdir_chain(precompute, "work_orders")
    _mkdir_chain(precompute, "member_receipts")
    lock_root = _safe_directory(
        expected_run / MEMBER_LOCK_DIRECTORY,
        parent=expected_run,
    )
    authorization = _authorization_document(
        root=Path(root),
        identity=identity,
        config=config,
        gate=matrix_gate,
    )
    authorization_path = precompute / "authorization.json"
    _publish_create_only(authorization_path, _canonical_bytes(authorization) + b"\n")
    work_orders: list[Path] = []
    for member in expected_stage09b_precompute_members():
        _mkdir_chain(precompute, str(Path(member.work_order_relative).parent))
        _mkdir_chain(precompute, str(Path(member.receipt_relative).parent))
        _mkdir_chain(lock_root, str(Path(member.lock_relative).parent))
        path = precompute / member.work_order_relative
        document = _work_order_document(
            root=repository,
            authorization_path=authorization_path,
            authorization=authorization,
            identity=identity,
            member=member,
        )
        _publish_create_only(path, _canonical_bytes(document) + b"\n")
        work_orders.append(path)
    _assert_exact_namespace(precompute, coordinator_optional=True)
    publication_guard()
    return authorization_path, tuple(work_orders)


def _validate_authorization(
    root: Path,
    path: Path,
) -> tuple[Mapping[str, Any], RunIdentity]:
    _require_safe_control_file(root, path, label="Stage-09b member authorization")
    value = _load_json(
        path, label="Stage-09b member authorization", canonical=True
    )
    keys = {
        "format", "status", "evidence_role", "final_scientific_authority",
        "required_final_authority", "stage", "run_identity", "resolved_config",
        "model_matrix", "matrix", "matrix_audit", "scientific_execution_contract",
    }
    document = _validate_self_hash(
        value, field="authorization_self_sha256", keys=keys,
        label="Stage-09b member authorization",
    )
    identity = _identity_from_mapping(document.get("run_identity"))
    if path != _precompute_root(root, identity.run_id) / "authorization.json":
        raise Stage09bPrecomputeError("authorization path is noncanonical")
    config = _validate_formal_config(document.get("resolved_config"))
    if sha256_json(config) != identity.config_sha256:
        raise Stage09bPrecomputeError("authorization config digest changed")
    matrix_binding = _validate_matrix_binding(document.get("model_matrix"))
    expected = _authorization_document(
        root=Path(root),
        identity=identity,
        config=config,
        gate=Stage09bMatrixGate(
            {}, {}, matrix_binding, str(matrix_binding["model_matrix_contract_id"])
        ),
    )
    if document != expected:
        raise Stage09bPrecomputeError("authorization content changed")
    return document, identity


def _validate_run_manifest(
    root: Path,
    identity: RunIdentity,
    config: object,
) -> None:
    expected = _run_directory(root, identity.run_id) / "run.json"
    _require_safe_control_file(root, expected, label="Stage-09b run manifest")
    manifest = _load_json(expected, label="Stage-09b run manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("identity") != identity.as_dict()
        or manifest.get("resolved_config") != config
    ):
        raise Stage09bPrecomputeError("Stage-09b run manifest differs")


def _validate_work_order_against_authorization(
    *, root: Path, path: Path, authorization_path: Path,
    authorization: Mapping[str, Any], identity: RunIdentity,
) -> Mapping[str, Any]:
    _require_safe_control_file(root, path, label="Stage-09b member work order")
    value = _load_json(
        path, label="Stage-09b member work order", canonical=True
    )
    keys = {
        "format", "status", "evidence_role", "final_scientific_authority",
        "required_final_authority", "stage", "authorization", "run_identity",
        "model_matrix_contract_id", "member", "scientific_member_config",
        "scientific_member_config_sha256", "artifacts", "execution_lock", "receipt",
        "publication_contract",
    }
    document = _validate_self_hash(
        value, field="work_order_self_sha256", keys=keys,
        label="Stage-09b member work order",
    )
    member = _member_from_mapping(document.get("member"))
    expected_path = _precompute_root(root, identity.run_id) / member.work_order_relative
    if path != expected_path:
        raise Stage09bPrecomputeError("member work-order path is noncanonical")
    expected = _work_order_document(
        root=root,
        authorization_path=authorization_path,
        authorization=authorization,
        identity=identity,
        member=member,
    )
    if document != expected:
        raise Stage09bPrecomputeError("member work-order content changed")
    return document


def validate_stage09b_member_work_order(
    *,
    root: str | Path,
    work_order: str | Path,
    expected_identity: RunIdentity | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> ValidatedStage09bWorkOrder:
    """Perform the complete live parent-equivalent gate in a raw worker."""
    repository = Path(root).resolve()
    path = Path(os.path.abspath(os.fspath(work_order)))
    # The run id/member cannot be trusted until the authorization is found from
    # the canonical path shape, so first require a strict repository-relative form.
    relative = _relative(repository, path, label="member work-order path")
    path = _safe_relative_file(repository, relative)
    parts = Path(relative).parts
    if len(parts) != 8 or parts[:3] != ("outputs", "runs", "09b_development_controls"):
        raise Stage09bPrecomputeError("member work-order path shape changed")
    run_id = parts[3]
    if parts[4:6] != (PRECOMPUTE_DIRECTORY, "work_orders"):
        # Arm/seed canonicality is checked from the decoded document below.
        raise Stage09bPrecomputeError("member work-order namespace changed")
    authorization_path = _precompute_root(repository, run_id) / "authorization.json"
    authorization, identity = _validate_authorization(repository, authorization_path)
    if expected_identity is not None and identity != expected_identity:
        raise Stage09bPrecomputeError("worker authorization uses another RunIdentity")
    if expected_config is not None and authorization.get("resolved_config") != expected_config:
        raise Stage09bPrecomputeError("worker authorization uses another config")
    document = _validate_work_order_against_authorization(
        root=repository,
        path=path,
        authorization_path=authorization_path,
        authorization=authorization,
        identity=identity,
    )
    assert_formal_numerical_policy(require_hash_randomization=True)
    closure = resolve_development_input_closure(repository)
    closure.assert_unchanged()
    if closure.binding_digest != identity.input_closure_sha256:
        raise Stage09bPrecomputeError("work order binds another input closure")
    if sha256_json(numerical_runtime_contract()) != identity.runtime_sha256:
        raise Stage09bPrecomputeError("work order binds another numerical runtime")
    canonical_panel = repository / "data_usgs" / "panel_usgs_120v2.parquet"
    canonical_registry = repository / "data_usgs" / "station_registry_v1.csv"
    if (
        sha256_file(canonical_panel) != identity.panel_sha256
        or sha256_file(canonical_registry) != identity.registry_sha256
    ):
        raise Stage09bPrecomputeError("work order binds another panel/registry")
    gate = validate_stage09b_model_matrix_gate(repository)
    if authorization.get("model_matrix") != gate.binding:
        raise Stage09bPrecomputeError("worker model-matrix gate differs from authorization")
    _validate_run_manifest(repository, identity, authorization.get("resolved_config"))
    validated = ValidatedStage09bWorkOrder(
        repository, path, document, authorization_path, authorization,
        identity, closure, gate,
    )
    validated.assert_unchanged()
    return validated


def execute_stage09b_member_work_orders(
    work_orders: Sequence[Path],
    *,
    workers: int,
    launch: Callable[[Path], object],
) -> None:
    """Execute exact work orders concurrently without changing their identity."""
    expected = expected_stage09b_precompute_members()
    paths = tuple(Path(path) for path in work_orders)
    if type(workers) is not int or not 1 <= workers <= MAX_PARALLEL_WORKERS:
        raise Stage09bPrecomputeError(
            f"execution-only worker count must be 1..{MAX_PARALLEL_WORKERS}"
        )
    if len(paths) != len(expected) or len(set(paths)) != len(expected):
        raise Stage09bPrecomputeError("launcher work-order registry is incomplete")
    precompute_roots = {path.parents[2] for path in paths}
    if len(precompute_roots) != 1:
        raise Stage09bPrecomputeError("launcher work orders do not share one namespace")
    precompute_root = next(iter(precompute_roots))
    expected_paths = {precompute_root / member.work_order_relative for member in expected}
    if set(paths) != expected_paths:
        raise Stage09bPrecomputeError("launcher has a missing or extra work order")
    failures: list[tuple[Path, Exception]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, path): path for path in paths}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # preserve every failed member identity
                failures.append((futures[future], exc))
    if failures:
        failed = ", ".join(path.as_posix() for path, _exc in sorted(failures))
        raise Stage09bPrecomputeError(f"Stage-09b workers failed: {failed}") from failures[0][1]


@contextmanager
def _component_safe_advisory_lock(
    *,
    root: Path,
    relative: str,
) -> Iterator[Path]:
    """Lock a file through no-follow directory descriptors for every component."""
    canonical = _normal_relative(relative, label="member execution lock")
    parts = Path(canonical).parts
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    opened_directories: list[int] = []
    lock_descriptor: int | None = None
    locked = False
    try:
        try:
            current = os.open(root.resolve(), directory_flags)
            opened_directories.append(current)
            for component in parts[:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                opened_directories.append(current)
        except OSError as exc:
            raise Stage09bPrecomputeError(
                "member execution-lock path contains a linked or unsafe directory"
            ) from exc
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(
                parts[-1],
                lock_flags,
                0o600,
                dir_fd=opened_directories[-1],
            )
        except OSError as exc:
            raise Stage09bPrecomputeError(
                "cannot safely open member execution lock"
            ) from exc
        lock_info = os.fstat(lock_descriptor)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or lock_info.st_nlink != 1
        ):
            raise Stage09bPrecomputeError(
                "member execution lock is not owner-private single-link state"
            )
        os.fchmod(lock_descriptor, 0o600)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        locked = True

        # Revalidate that the descriptor chain is still the lexical repository
        # chain after the blocking flock acquisition.  This closes the practical
        # freeze-to-worker symlink swap window without ever following a link.
        lexical = root.resolve() / canonical
        try:
            parent_info = lexical.parent.lstat()
            final_info = lexical.lstat()
        except OSError as exc:
            raise Stage09bPrecomputeError(
                "member execution-lock namespace changed during acquisition"
            ) from exc
        opened_parent = os.fstat(opened_directories[-1])
        if (
            stat.S_ISLNK(parent_info.st_mode)
            or not stat.S_ISDIR(parent_info.st_mode)
            or (parent_info.st_dev, parent_info.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
            or stat.S_ISLNK(final_info.st_mode)
            or not stat.S_ISREG(final_info.st_mode)
            or (final_info.st_dev, final_info.st_ino)
            != (lock_info.st_dev, lock_info.st_ino)
            or final_info.st_nlink != 1
        ):
            raise Stage09bPrecomputeError(
                "member execution-lock namespace changed during acquisition"
            )
        yield lexical
    finally:
        if lock_descriptor is not None:
            try:
                if locked:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            finally:
                os.close(lock_descriptor)
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


@contextmanager
def stage09b_member_execution_lock(
    validated: ValidatedStage09bWorkOrder,
) -> Iterator[Path]:
    """Serialize every direct invocation of one deterministic member namespace."""
    relative = validated.document.get("execution_lock")
    if type(relative) is not str:
        raise Stage09bPrecomputeError("member execution-lock path is malformed")
    expected = _member_lock_relative(validated.identity, validated.member)
    if relative != expected:
        raise Stage09bPrecomputeError("member execution-lock path changed")
    expected_parent = (
        validated.run_directory / MEMBER_LOCK_DIRECTORY
        / f"arm{validated.member.arm_index:02d}"
    )
    lock_path = validated.root / relative
    if lock_path.parent != expected_parent:
        raise Stage09bPrecomputeError("member execution-lock namespace is unsafe")
    with _component_safe_advisory_lock(
        root=validated.root,
        relative=relative,
    ) as acquired:
        validated.assert_unchanged()
        yield acquired
        validated.assert_unchanged()


def _semantic_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "prediction_content_sha256", "forecast_key_truth_sha256",
        "prediction_rows", "checkpoint_exact_replay_sha256",
    }:
        raise Stage09bPrecomputeError("member semantic evidence schema changed")
    for field in (
        "prediction_content_sha256", "forecast_key_truth_sha256",
        "checkpoint_exact_replay_sha256",
    ):
        if type(value.get(field)) is not str or _DIGEST.fullmatch(value[field]) is None:
            raise Stage09bPrecomputeError("member semantic digest is malformed")
    if (
        type(value.get("prediction_rows")) is not int
        or int(value["prediction_rows"]) < 1
        or value["prediction_content_sha256"]
        != value["checkpoint_exact_replay_sha256"]
    ):
        raise Stage09bPrecomputeError("member was not proved by exact checkpoint replay")
    return dict(value)


def publish_stage09b_member_receipt(
    validated: ValidatedStage09bWorkOrder,
    *,
    semantic_evidence: Mapping[str, Any],
    exact_replay_verified: bool,
    publication_guard: Callable[[], object],
) -> Path:
    """Commit one member create-only after a fresh exact checkpoint replay."""
    if exact_replay_verified is not True:
        raise Stage09bPrecomputeError("member reuse requires fresh exact replay")
    semantic = _semantic_evidence(semantic_evidence)
    publication_guard()
    validated.assert_unchanged()
    artifacts = validated.artifact_paths
    bindings = {
        label: _file_binding(
            validated.root,
            _relative(validated.root, path, label=f"member {label}"),
        )
        for label, path in artifacts.items()
    }
    stable = {
        "format": MEMBER_RECEIPT_FORMAT,
        "status": "COMMITTED_INTERMEDIATE_AFTER_PRODUCER_CHECKPOINT_REPLAY",
        "evidence_role": _INTERMEDIATE_EVIDENCE_ROLE,
        "final_scientific_authority": False,
        "required_final_authority": "outputs/models/route_a_stage09b_completion.json",
        "stage": "09b_development_controls",
        "authorization": _file_binding(
            validated.root,
            _relative(validated.root, validated.authorization_path, label="authorization"),
        ),
        "work_order": _file_binding(
            validated.root,
            _relative(validated.root, validated.path, label="work order"),
        ),
        "run_identity": validated.identity.as_dict(),
        "input_closure_sha256": validated.identity.input_closure_sha256,
        "source_sha256": validated.identity.source_sha256,
        "runtime_sha256": validated.identity.runtime_sha256,
        "model_matrix": dict(validated.matrix_gate.binding),
        "member": validated.member.as_dict(),
        "scientific_member_config": dict(validated.document["scientific_member_config"]),
        "scientific_member_config_sha256": validated.document[
            "scientific_member_config_sha256"
        ],
        "artifacts": bindings,
        "semantic_evidence": semantic,
        "reuse_contract": {
            "exact_checkpoint_replay_performed_this_invocation": True,
            "existing_receipt_may_be_reused_only_if_all_bytes_remain_exact": True,
        },
    }
    receipt = _with_self_hash(stable, "receipt_self_sha256")
    payload = _canonical_bytes(receipt) + b"\n"
    _publish_create_only(validated.receipt_path, payload)
    publication_guard()
    validated.assert_unchanged()
    if _load_json(
        validated.receipt_path, label="member receipt", canonical=True
    ) != receipt:
        raise Stage09bPrecomputeError("published member receipt changed")
    observed_member, _prediction, observed_semantic = _validate_member_receipt(
        root=validated.root,
        path=validated.receipt_path,
        work_order=validated.document,
        authorization=validated.authorization,
        identity=validated.identity,
    )
    if observed_member != validated.member or observed_semantic != semantic:
        raise Stage09bPrecomputeError("published member receipt replay changed")
    return validated.receipt_path


def _validate_member_receipt(
    *, root: Path, path: Path, work_order: Mapping[str, Any],
    authorization: Mapping[str, Any], identity: RunIdentity,
) -> tuple[Stage09bMember, Path, Mapping[str, Any]]:
    _require_safe_control_file(root, path, label="Stage-09b member receipt")
    value = _load_json(path, label="Stage-09b member receipt", canonical=True)
    keys = {
        "format", "status", "evidence_role", "final_scientific_authority",
        "required_final_authority", "stage", "authorization", "work_order",
        "run_identity", "input_closure_sha256", "source_sha256", "runtime_sha256",
        "model_matrix", "member", "scientific_member_config",
        "scientific_member_config_sha256", "artifacts", "semantic_evidence",
        "reuse_contract",
    }
    document = _validate_self_hash(
        value, field="receipt_self_sha256", keys=keys,
        label="Stage-09b member receipt",
    )
    member = _member_from_mapping(document.get("member"))
    expected_path = _precompute_root(root, identity.run_id) / member.receipt_relative
    if path != expected_path:
        raise Stage09bPrecomputeError("member receipt path is noncanonical")
    artifacts = work_order.get("artifacts")
    bindings = document.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(bindings, Mapping):
        raise Stage09bPrecomputeError("member artifact registry is malformed")
    if set(bindings) != set(artifacts):
        raise Stage09bPrecomputeError("member artifact binding registry changed")
    for label, relative in artifacts.items():
        resolved = _assert_file_binding(root, bindings[label], label=f"member {label}")
        if resolved != _safe_relative_file(root, str(relative)):
            raise Stage09bPrecomputeError("member artifact path changed")
    semantic = _semantic_evidence(document.get("semantic_evidence"))
    expected_static = {
        "format": MEMBER_RECEIPT_FORMAT,
        "status": "COMMITTED_INTERMEDIATE_AFTER_PRODUCER_CHECKPOINT_REPLAY",
        "evidence_role": _INTERMEDIATE_EVIDENCE_ROLE,
        "final_scientific_authority": False,
        "required_final_authority": "outputs/models/route_a_stage09b_completion.json",
        "stage": "09b_development_controls",
        "authorization": work_order["authorization"],
        "work_order": _file_binding(
            root,
            _relative(
                root,
                _precompute_root(root, identity.run_id) / member.work_order_relative,
                label="work order",
            ),
        ),
        "run_identity": identity.as_dict(),
        "input_closure_sha256": identity.input_closure_sha256,
        "source_sha256": identity.source_sha256,
        "runtime_sha256": identity.runtime_sha256,
        "model_matrix": authorization["model_matrix"],
        "member": member.as_dict(),
        "scientific_member_config": work_order["scientific_member_config"],
        "scientific_member_config_sha256": work_order[
            "scientific_member_config_sha256"
        ],
        "artifacts": dict(bindings),
        "semantic_evidence": semantic,
        "reuse_contract": {
            "exact_checkpoint_replay_performed_this_invocation": True,
            "existing_receipt_may_be_reused_only_if_all_bytes_remain_exact": True,
        },
    }
    if document != _with_self_hash(expected_static, "receipt_self_sha256"):
        raise Stage09bPrecomputeError("member receipt contract changed")
    prediction = _assert_file_binding(
        root, bindings["prediction"], label="member prediction"
    )
    return member, prediction, semantic


def _assert_exact_namespace(precompute: Path, *, coordinator_optional: bool) -> None:
    expected_work_orders = {
        member.work_order_relative for member in expected_stage09b_precompute_members()
    }
    expected_receipts = {
        member.receipt_relative for member in expected_stage09b_precompute_members()
    }
    pending_receipt_temps: set[Path] = set()
    if coordinator_optional:
        for relative in expected_receipts:
            pending_receipt_temps.update(
                _validated_pending_create_only_temps(precompute / relative)
            )
    for directory_name, expected, complete in (
        ("work_orders", expected_work_orders, True),
        ("member_receipts", expected_receipts, not coordinator_optional),
    ):
        directory = precompute / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise Stage09bPrecomputeError(f"{directory_name} namespace is unsafe")
        observed: set[str] = set()
        expected_directories = {
            (precompute / relative).parent.relative_to(precompute).as_posix()
            for relative in expected
        }
        expected_directories.add(directory_name)
        observed_directories = {directory_name}
        for path in directory.rglob("*"):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise Stage09bPrecomputeError(f"unsafe {directory_name} namespace entry")
            if path.is_dir():
                observed_directories.add(path.relative_to(precompute).as_posix())
            else:
                if directory_name == "member_receipts" and path in pending_receipt_temps:
                    continue
                observed.add(path.relative_to(precompute).as_posix())
        if observed_directories != expected_directories:
            missing = sorted(expected_directories - observed_directories)
            extra = sorted(observed_directories - expected_directories)
            raise Stage09bPrecomputeError(
                f"{directory_name} directory closure differs: "
                f"missing={missing}, extra={extra}"
            )
        if complete and observed != expected:
            missing, extra = sorted(expected - observed), sorted(observed - expected)
            raise Stage09bPrecomputeError(
                f"{directory_name} closure differs: missing={missing}, extra={extra}"
            )
        if not complete and not observed.issubset(expected):
            raise Stage09bPrecomputeError(f"{directory_name} contains an extra member")
    allowed_top = {
        "authorization.json", "work_orders", "member_receipts",
        "coordinator_receipt.json",
    }
    pending_coordinator_temps = _validated_pending_create_only_temps(
        precompute / "coordinator_receipt.json"
    )
    observed_top = {
        path.name for path in precompute.iterdir()
        if path not in pending_coordinator_temps
    }
    if not observed_top.issubset(allowed_top) or "authorization.json" not in observed_top:
        raise Stage09bPrecomputeError("precompute namespace contains an unexpected entry")


def finalize_stage09b_precompute(
    *,
    root: str | Path,
    authorization_path: str | Path,
    semantic_inspector: Callable[[Stage09bMember, Path], Mapping[str, Any]],
    publication_guard: Callable[[], object],
) -> tuple[dict[tuple[str, int], Path], Path]:
    """Recheck prediction semantics and freeze the exact intermediate closure."""
    repository = Path(root).resolve()
    authorization_file = Path(os.path.abspath(os.fspath(authorization_path)))
    authorization, identity = _validate_authorization(repository, authorization_file)
    config = _validate_formal_config(authorization["resolved_config"])
    assert_formal_numerical_policy(require_hash_randomization=True)
    closure = resolve_development_input_closure(repository)
    closure.assert_unchanged()
    if (
        closure.binding_digest != identity.input_closure_sha256
        or sha256_json(numerical_runtime_contract()) != identity.runtime_sha256
    ):
        raise Stage09bPrecomputeError("coordinator live scientific identity changed")
    gate = validate_stage09b_model_matrix_gate(repository)
    if authorization.get("model_matrix") != gate.binding:
        raise Stage09bPrecomputeError("coordinator model-matrix gate changed")
    _validate_run_manifest(repository, identity, config)
    precompute = _precompute_root(repository, identity.run_id)
    _assert_exact_namespace(precompute, coordinator_optional=False)
    resolved: dict[tuple[str, int], Path] = {}
    receipt_bindings: list[dict[str, Any]] = []
    common_registry_digest: str | None = None
    common_rows: int | None = None
    for member in expected_stage09b_precompute_members():
        work_order_path = precompute / member.work_order_relative
        work_order = _validate_work_order_against_authorization(
            root=repository,
            path=work_order_path,
            authorization_path=authorization_file,
            authorization=authorization,
            identity=identity,
        )
        receipt_path = precompute / member.receipt_relative
        observed_member, prediction, semantic = _validate_member_receipt(
            root=repository,
            path=receipt_path,
            work_order=work_order,
            authorization=authorization,
            identity=identity,
        )
        if observed_member != member:
            raise Stage09bPrecomputeError("member receipt order/identity changed")
        independently_replayed = _semantic_evidence(
            semantic_inspector(member, prediction)
        )
        if semantic != independently_replayed:
            raise Stage09bPrecomputeError(
                f"{member.member_id} receipt differs from independent prediction replay"
            )
        registry_digest = semantic["forecast_key_truth_sha256"]
        rows = int(semantic["prediction_rows"])
        if common_registry_digest is None:
            common_registry_digest, common_rows = registry_digest, rows
        elif registry_digest != common_registry_digest or rows != common_rows:
            raise Stage09bPrecomputeError(
                "Stage-09b members do not share exact forecast keys and y_true"
            )
        resolved[(member.arm_id, member.seed)] = prediction
        receipt_bindings.append(
            {**member.as_dict(), "receipt": _file_binding(
                repository, _relative(repository, receipt_path, label="member receipt")
            )}
        )
    expected_pairs = _expected_comparisons()
    member_keys = set(resolved)
    for comparison in expected_pairs:
        for seed in comparison["seeds"]:
            if (
                (str(comparison["candidate_arm_id"]), int(seed)) not in member_keys
                or (str(comparison["reference_arm_id"]), int(seed)) not in member_keys
            ):
                raise Stage09bPrecomputeError("same-seed paired comparison is incomplete")
    if len(resolved) != EXPECTED_MEMBER_COUNT or len(member_keys) != EXPECTED_MEMBER_COUNT:
        raise Stage09bPrecomputeError("coordinator member registry is not exact 45")
    assert common_registry_digest is not None and common_rows is not None
    stable = {
        "format": COORDINATOR_RECEIPT_FORMAT,
        "status": "PASS_INTERMEDIATE_45_MEMBER_SCHEDULING_CLOSURE",
        "evidence_role": _INTERMEDIATE_EVIDENCE_ROLE,
        "final_scientific_authority": False,
        "required_final_authority": "outputs/models/route_a_stage09b_completion.json",
        "stage": "09b_development_controls",
        "authorization": _file_binding(
            repository, _relative(repository, authorization_file, label="authorization")
        ),
        "run_identity": identity.as_dict(),
        "model_matrix": dict(gate.binding),
        "member_receipts": receipt_bindings,
        "matrix_audit": {
            "expected_members": EXPECTED_MEMBER_COUNT,
            "observed_members": len(resolved),
            "missing_members": 0,
            "duplicate_members": 0,
            "extra_members": 0,
            "common_forecast_keys_and_y_true_sha256": common_registry_digest,
            "prediction_rows_per_member": common_rows,
            "same_seed_paired_comparison_count": len(expected_pairs),
            "same_seed_pairs": expected_pairs,
        },
        "information_fairness": {
            "plain_arms": list(_PLAIN_ARMS),
            "allowed_consumed_batch_keys": list(PLAIN_CONTROL_ALLOWED_INPUTS),
            "explicitly_forbidden_batch_keys": list(PLAIN_CONTROL_FORBIDDEN_INPUTS),
            "common_anchor": "damped_prior",
            "prediction_head": "unrestricted_residual_added_to_damped_prior",
            "parameter_budget_within_2pct": True,
            "producer_replay_declarations_semantically_consistent": True,
            "full_training_trajectory_replay_verified_here": False,
        },
        "scientific_equivalence": {
            "declared_same_arm_seed_config_epoch_and_early_stopping": True,
            "declared_same_training_checkpoint_and_prediction_functions": True,
            "parallelism_changes_only_execution_schedule": True,
            "worker_count_or_schedule_in_run_identity": False,
            "independent_checkpoint_replay_deferred_to_final_authority": True,
        },
    }
    coordinator = _with_self_hash(stable, "coordinator_self_sha256")
    coordinator_path = precompute / "coordinator_receipt.json"
    publication_guard()
    gate.assert_bytes_unchanged(repository)
    _publish_create_only(coordinator_path, _canonical_bytes(coordinator) + b"\n")
    publication_guard()
    if _load_json(
        coordinator_path, label="coordinator receipt", canonical=True
    ) != coordinator:
        raise Stage09bPrecomputeError("coordinator receipt changed after publication")
    return resolved, coordinator_path


__all__ = [
    "AUTHORIZATION_FORMAT",
    "COORDINATOR_RECEIPT_FORMAT",
    "EXPECTED_MEMBER_COUNT",
    "MAX_PARALLEL_WORKERS",
    "MEMBER_LOCK_DIRECTORY",
    "MEMBER_RECEIPT_FORMAT",
    "PRECOMPUTE_DIRECTORY",
    "Stage09bMatrixGate",
    "Stage09bMember",
    "Stage09bPrecomputeError",
    "ValidatedStage09bWorkOrder",
    "WORK_ORDER_FORMAT",
    "allocate_create_only_artifact_temp",
    "execute_stage09b_member_work_orders",
    "expected_stage09b_precompute_members",
    "finalize_stage09b_precompute",
    "freeze_stage09b_precompute_plan",
    "publish_create_only_artifact_from_temp",
    "publish_stage09b_member_receipt",
    "recover_create_only_artifact_state",
    "stage09b_member_execution_lock",
    "validate_stage09b_member_work_order",
    "validate_stage09b_model_matrix_gate",
]
