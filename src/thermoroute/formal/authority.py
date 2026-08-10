"""Inert primitives for binding governance data to repository files.

This module does not execute work, interpret a scientific protocol, or grant a
domain-specific permission.  Its private construction tokens enforce ordinary
Python API invariants only.  They are not secrets and are not a security
boundary against hostile code running in this process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Self, cast


class FormalAuthorityError(RuntimeError):
    """Base exception for the shared formal-governance primitives."""


class PathPolicyError(FormalAuthorityError):
    """Raised when a root or repository-relative path violates policy."""


class CaptureError(FormalAuthorityError):
    """Raised when a stable regular-file capture cannot be made."""


class PlatformSupportError(CaptureError):
    """Raised when required fail-closed POSIX filesystem operations are absent."""


class FileTypeError(CaptureError):
    """Raised when a capture path is not a regular file under real directories."""


class SymlinkRejectedError(FileTypeError):
    """Raised when the root or a relative-path component is a symbolic link."""


class HardlinkRejectedError(FileTypeError):
    """Raised when a regular file has more than one filesystem link."""


class UnstableCaptureError(CaptureError):
    """Raised when file or directory identity changes while it is captured."""


class BindingValidationError(FormalAuthorityError):
    """Raised when a :class:`Binding` is internally inconsistent."""


class SnapshotMismatchError(FormalAuthorityError):
    """Raised when recaptured bytes no longer equal a retained snapshot."""


class CanonicalJSONError(FormalAuthorityError, ValueError):
    """Raised when JSON is invalid, non-finite, or not in canonical form."""


class DuplicateJSONKeyError(CanonicalJSONError):
    """Raised when any JSON object repeats a key."""


class JSONFieldError(CanonicalJSONError):
    """Raised when an object does not have its exact closed field set."""


class AuthorityError(FormalAuthorityError):
    """Base exception for authority construction and validation failures."""


class AuthorityConstructionError(AuthorityError, TypeError):
    """Raised when code attempts to bypass an authority factory."""


class AuthorityValidationError(AuthorityError):
    """Raised when an authority object fails structural or seal validation."""


class AuthorityMismatchError(AuthorityValidationError):
    """Raised when an authority has the wrong action or profile."""


# Hardlinks are rejected rather than conditionally accepted.  This is kept as a
# named policy so consumers do not infer a caller-controlled exception.
HARDLINK_POLICY: Final[str] = "reject-st_nlink-not-one"

_READ_CHUNK_SIZE: Final[int] = 1024 * 1024
_SHA256_HEX_LENGTH: Final[int] = 64
_PENDING_ISSUER: Final[object] = object()
_EXECUTION_ISSUER: Final[object] = object()
_CONSTRUCTION_TOKEN: Final[object] = object()
_INTERNAL_EXECUTION_FACTORY_TOKEN: Final[object] = object()


def _validate_relative_path(path: object) -> PurePosixPath:
    if type(path) is not PurePosixPath:
        raise PathPolicyError("capture path must be an exact PurePosixPath")
    if path.is_absolute() or path.anchor:
        raise PathPolicyError(f"capture path must be repository-relative: {path!s}")
    if not path.parts:
        raise PathPolicyError("capture path must name a file, not the repository root")
    for part in path.parts:
        if (
            not part
            or part in {".", ".."}
            or "\x00" in part
            or "/" in part
            or PurePosixPath(part).name != part
        ):
            raise PathPolicyError(f"capture path has a non-canonical component: {part!r}")
    if path.as_posix() != "/".join(path.parts):
        raise PathPolicyError(f"capture path is not canonical POSIX syntax: {path!s}")
    return path


def _validate_digest(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BindingValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class Binding:
    """Retained bytes and their repository-relative content identity.

    A directly constructed ``Binding`` proves internal consistency, not that the
    bytes came from a filesystem capture.  Use :func:`capture_binding` when that
    provenance matters.
    """

    path: PurePosixPath
    sha256: str
    size: int
    content: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _validate_binding(self)


def _validate_binding(value: object) -> Binding:
    if type(value) is not Binding:
        raise BindingValidationError("snapshot members must be exact Binding instances")
    path = _validate_relative_path(value.path)
    digest = _validate_digest(value.sha256, label=f"binding {path!s} sha256")
    if type(value.size) is not int or value.size < 0:
        raise BindingValidationError(f"binding {path!s} size must be a non-negative integer")
    if type(value.content) is not bytes:
        raise BindingValidationError(f"binding {path!s} content must be exact bytes")
    if value.size != len(value.content):
        raise BindingValidationError(f"binding {path!s} retained-byte size does not match")
    actual_digest = hashlib.sha256(value.content).hexdigest()
    if not hmac.compare_digest(digest, actual_digest):
        raise BindingValidationError(f"binding {path!s} retained-byte digest does not match")
    return value


def _require_capture_platform() -> None:
    if os.name != "posix":
        raise PlatformSupportError("stable rooted capture requires POSIX openat semantics")
    missing_flags = [
        name
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        if not hasattr(os, name)
    ]
    if missing_flags:
        raise PlatformSupportError(
            "platform lacks required fail-closed flags: " + ", ".join(missing_flags)
        )
    if (
        os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise PlatformSupportError("platform lacks required dir_fd/stat no-follow support")


def _root_path(root: str | os.PathLike[str]) -> Path:
    try:
        path = Path(root)
    except (TypeError, ValueError) as exc:
        raise PathPolicyError("repository root must be a filesystem path") from exc
    if not path.is_absolute():
        raise PathPolicyError("repository root must be absolute")
    if path.anchor != os.sep:
        raise PathPolicyError("repository root must use one canonical POSIX root anchor")
    if ".." in path.parts or "\x00" in os.fspath(path):
        raise PathPolicyError("repository root must not contain non-canonical components")
    return path


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and stat.S_IFMT(first.st_mode) == stat.S_IFMT(second.st_mode)
    )


def _state_tuple(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_regular(value: os.stat_result, *, label: str) -> None:
    if stat.S_ISLNK(value.st_mode):
        raise SymlinkRejectedError(f"symbolic link rejected: {label}")
    if not stat.S_ISREG(value.st_mode):
        raise FileTypeError(f"regular file required: {label}")
    if value.st_nlink != 1:
        raise HardlinkRejectedError(
            f"hardlinked file rejected by {HARDLINK_POLICY}: {label} has {value.st_nlink} links"
        )


def _require_directory(value: os.stat_result, *, label: str) -> None:
    if stat.S_ISLNK(value.st_mode):
        raise SymlinkRejectedError(f"symbolic link rejected: {label}")
    if not stat.S_ISDIR(value.st_mode):
        raise FileTypeError(f"directory component required: {label}")


def _entry_stat(parent_fd: int, name: str, *, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise CaptureError(f"cannot inspect {label}: {exc.strerror or exc}") from exc


def _post_entry_stat(parent_fd: int, name: str, *, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise UnstableCaptureError(f"directory entry changed during capture: {label}") from exc


_DirectoryRecord = tuple[int, str, int, os.stat_result, str]


@dataclass(frozen=True, slots=True)
class _RootAnchor:
    capture_fd: int
    filesystem_fd: int
    filesystem_stat: os.stat_result
    directories: tuple[_DirectoryRecord, ...]


def _open_filesystem_root() -> tuple[int, os.stat_result]:
    try:
        entry_before = os.lstat("/")
    except OSError as exc:
        raise CaptureError(f"cannot inspect filesystem root: {exc.strerror or exc}") from exc
    _require_directory(entry_before, label="filesystem root /")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = os.open("/", flags)
    except OSError as exc:
        raise CaptureError(f"cannot anchor filesystem root: {exc.strerror or exc}") from exc
    try:
        opened = os.fstat(root_fd)
        entry_after = os.lstat("/")
        _require_directory(opened, label="filesystem root /")
        _require_directory(entry_after, label="filesystem root /")
        if not (_same_identity(entry_before, opened) and _same_identity(opened, entry_after)):
            raise UnstableCaptureError("filesystem root identity changed while opening")
    except BaseException:
        os.close(root_fd)
        raise
    return root_fd, opened


def _open_root(root: Path) -> _RootAnchor:
    """Anchor ``root`` by walking every absolute component from ``/``."""

    filesystem_fd, filesystem_stat = _open_filesystem_root()
    current_fd = filesystem_fd
    directories: list[_DirectoryRecord] = []
    try:
        for index, component in enumerate(root.parts[1:], start=1):
            label = "/" + "/".join(root.parts[1 : index + 1])
            directory_fd, opened = _open_directory(current_fd, component, label=label)
            directories.append((current_fd, component, directory_fd, opened, label))
            current_fd = directory_fd
        return _RootAnchor(
            capture_fd=current_fd,
            filesystem_fd=filesystem_fd,
            filesystem_stat=filesystem_stat,
            directories=tuple(directories),
        )
    except BaseException:
        for _, _, directory_fd, _, _ in reversed(directories):
            os.close(directory_fd)
        os.close(filesystem_fd)
        raise


def _verify_root(anchor: _RootAnchor) -> None:
    _verify_directories(anchor.directories)
    try:
        current = os.fstat(anchor.filesystem_fd)
        entry = os.lstat("/")
    except OSError as exc:
        raise UnstableCaptureError("filesystem root changed during capture") from exc
    _require_directory(current, label="filesystem root /")
    _require_directory(entry, label="filesystem root /")
    if not (_same_identity(anchor.filesystem_stat, current) and _same_identity(current, entry)):
        raise UnstableCaptureError("filesystem root identity changed during capture")


def _close_root(anchor: _RootAnchor) -> None:
    for _, _, directory_fd, _, _ in reversed(anchor.directories):
        os.close(directory_fd)
    os.close(anchor.filesystem_fd)


def _open_directory(parent_fd: int, name: str, *, label: str) -> tuple[int, os.stat_result]:
    entry_before = _entry_stat(parent_fd, name, label=label)
    _require_directory(entry_before, label=label)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CaptureError(
            f"cannot open directory component {label}: {exc.strerror or exc}"
        ) from exc
    try:
        opened = os.fstat(directory_fd)
        entry_after = _post_entry_stat(parent_fd, name, label=label)
        _require_directory(opened, label=label)
        _require_directory(entry_after, label=label)
        if not (_same_identity(entry_before, opened) and _same_identity(opened, entry_after)):
            raise UnstableCaptureError(f"directory identity changed while opening: {label}")
    except BaseException:
        os.close(directory_fd)
        raise
    return directory_fd, opened


def _read_regular_file(parent_fd: int, name: str, *, label: str) -> bytes:
    entry_before = _entry_stat(parent_fd, name, label=label)
    _require_regular(entry_before, label=label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        file_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise CaptureError(f"cannot open regular file {label}: {exc.strerror or exc}") from exc
    try:
        opened = os.fstat(file_fd)
        _require_regular(opened, label=label)
        if _state_tuple(entry_before) != _state_tuple(opened):
            raise UnstableCaptureError(f"file changed while opening: {label}")

        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, _READ_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)

        after_read = os.fstat(file_fd)
        entry_after = _post_entry_stat(parent_fd, name, label=label)
        _require_regular(after_read, label=label)
        _require_regular(entry_after, label=label)
        if _state_tuple(opened) != _state_tuple(after_read):
            raise UnstableCaptureError(f"file metadata changed while reading: {label}")
        if _state_tuple(after_read) != _state_tuple(entry_after):
            raise UnstableCaptureError(f"file directory entry changed while reading: {label}")
        if len(content) != after_read.st_size:
            raise UnstableCaptureError(f"file byte count changed while reading: {label}")
        return content
    finally:
        os.close(file_fd)


def _verify_directories(
    directories: Sequence[_DirectoryRecord],
) -> None:
    for parent_fd, name, directory_fd, opened, label in reversed(directories):
        try:
            current = os.fstat(directory_fd)
            entry = _post_entry_stat(parent_fd, name, label=label)
        except OSError as exc:
            raise UnstableCaptureError(f"directory changed during capture: {label}") from exc
        _require_directory(current, label=label)
        _require_directory(entry, label=label)
        if not (_same_identity(opened, current) and _same_identity(current, entry)):
            raise UnstableCaptureError(f"directory identity changed during capture: {label}")


def _capture_from_root(root_fd: int, relative_path: PurePosixPath) -> Binding:
    current_fd = root_fd
    opened_directories: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        for index, component in enumerate(relative_path.parts[:-1], start=1):
            label = "/".join(relative_path.parts[:index])
            directory_fd, opened = _open_directory(current_fd, component, label=label)
            opened_directories.append((current_fd, component, directory_fd, opened, label))
            current_fd = directory_fd
        content = _read_regular_file(
            current_fd,
            relative_path.parts[-1],
            label=relative_path.as_posix(),
        )
        _verify_directories(opened_directories)
        return Binding(
            path=relative_path,
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
            content=content,
        )
    finally:
        for _, _, directory_fd, _, _ in reversed(opened_directories):
            os.close(directory_fd)


def capture_binding(root: str | os.PathLike[str], relative_path: PurePosixPath) -> Binding:
    """Capture one stable, non-symlinked, singly linked regular file.

    The absolute root is anchored by a directory descriptor.  Every relative
    component is traversed with ``dir_fd``/``O_NOFOLLOW`` and retained until
    post-read identity checks complete.  This closes ordinary pathname races;
    it is not a hostile-code sandbox.
    """

    _require_capture_platform()
    canonical_path = _validate_relative_path(relative_path)
    canonical_root = _root_path(root)
    anchor = _open_root(canonical_root)
    try:
        binding = _capture_from_root(anchor.capture_fd, canonical_path)
        _verify_root(anchor)
        return binding
    finally:
        _close_root(anchor)


def _snapshot_paths(paths: Sequence[PurePosixPath]) -> tuple[PurePosixPath, ...]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise PathPolicyError("snapshot paths must be a finite sequence of PurePosixPath values")
    canonical = tuple(_validate_relative_path(path) for path in paths)
    if not canonical:
        raise PathPolicyError("snapshot must contain at least one path")
    if len(set(canonical)) != len(canonical):
        raise PathPolicyError("snapshot paths must be unique")
    return tuple(sorted(canonical, key=PurePosixPath.as_posix))


def capture_snapshot(
    root: str | os.PathLike[str], paths: Sequence[PurePosixPath]
) -> tuple[Binding, ...]:
    """Capture a canonical path-ordered tuple of bindings under one root fd."""

    _require_capture_platform()
    canonical_paths = _snapshot_paths(paths)
    canonical_root = _root_path(root)
    anchor = _open_root(canonical_root)
    try:
        bindings = tuple(_capture_from_root(anchor.capture_fd, path) for path in canonical_paths)
        _verify_root(anchor)
        return bindings
    finally:
        _close_root(anchor)


def _validate_snapshot(snapshot: object) -> tuple[Binding, ...]:
    if type(snapshot) is not tuple or not snapshot:
        raise BindingValidationError("snapshot must be a non-empty exact tuple of bindings")
    bindings = tuple(_validate_binding(binding) for binding in snapshot)
    paths = tuple(binding.path for binding in bindings)
    if len(set(paths)) != len(paths):
        raise BindingValidationError("snapshot binding paths must be unique")
    if paths != tuple(sorted(paths, key=PurePosixPath.as_posix)):
        raise BindingValidationError("snapshot bindings must be in canonical path order")
    return bindings


def recapture_snapshot(
    root: str | os.PathLike[str], snapshot: tuple[Binding, ...]
) -> tuple[Binding, ...]:
    """Capture the current bytes for every path in a validated snapshot."""

    bindings = _validate_snapshot(snapshot)
    return capture_snapshot(root, tuple(binding.path for binding in bindings))


def revalidate_snapshot(root: str | os.PathLike[str], snapshot: tuple[Binding, ...]) -> None:
    """Require a fresh rooted capture to equal all retained snapshot bytes."""

    expected = _validate_snapshot(snapshot)
    current = recapture_snapshot(root, expected)
    for old, new in zip(expected, current, strict=True):
        if old != new:
            raise SnapshotMismatchError(
                f"snapshot content changed for {old.path!s}: "
                f"expected {old.sha256}/{old.size}, found {new.sha256}/{new.size}"
            )


def _copy_json(value: object, *, frozen: bool, active: set[int]) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return value
    if value_type is str:
        string_value = cast(str, value)
        if any(0xD800 <= ord(character) <= 0xDFFF for character in string_value):
            raise CanonicalJSONError("JSON strings must contain only Unicode scalar values")
        return string_value
    if value_type is float:
        float_value = cast(float, value)
        if not math.isfinite(float_value):
            raise CanonicalJSONError("JSON numbers must be finite")
        return float_value
    if value_type in {dict, MappingProxyType}:
        mapping_value = cast(Mapping[object, object], value)
        identity = id(value)
        if identity in active:
            raise CanonicalJSONError("cyclic JSON object")
        active.add(identity)
        try:
            items: list[tuple[str, Any]] = []
            for key, item in mapping_value.items():
                if type(key) is not str:
                    raise CanonicalJSONError("JSON object keys must be exact strings")
                if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                    raise CanonicalJSONError(
                        "JSON object keys must contain only Unicode scalar values"
                    )
                items.append((key, _copy_json(item, frozen=frozen, active=active)))
            items.sort(key=lambda pair: pair[0])
            copied = dict(items)
            return MappingProxyType(copied) if frozen else copied
        finally:
            active.remove(identity)
    if value_type in {list, tuple}:
        sequence_value = cast(Sequence[object], value)
        identity = id(value)
        if identity in active:
            raise CanonicalJSONError("cyclic JSON array")
        active.add(identity)
        try:
            copied_items = tuple(
                _copy_json(item, frozen=frozen, active=active) for item in sequence_value
            )
            return copied_items if frozen else list(copied_items)
        finally:
            active.remove(identity)
    raise CanonicalJSONError(f"unsupported JSON value type: {value_type.__name__}")


def freeze_json(value: object) -> object:
    """Return a defensive, recursively immutable copy of a JSON value."""

    return _copy_json(value, frozen=True, active=set())


def canonical_json_bytes(value: object) -> bytes:
    """Serialize finite JSON as sorted compact ASCII plus one trailing newline."""

    plain = _copy_json(value, frozen=False, active=set())
    try:
        rendered = json.dumps(
            plain,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (rendered + "\n").encode("ascii", errors="strict")
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError) as exc:
        raise CanonicalJSONError("value cannot be serialized as canonical JSON") from exc


def parse_canonical_json_object(payload: bytes, *, label: str = "document") -> Mapping[str, object]:
    """Parse one duplicate-free canonical JSON object into immutable values."""

    if type(payload) is not bytes:
        raise CanonicalJSONError(f"{label} payload must be exact bytes")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise CanonicalJSONError(f"{label} must not contain a UTF-8 BOM")

    def reject_constant(value: str) -> None:
        raise CanonicalJSONError(f"{label} contains non-finite constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise CanonicalJSONError(f"{label} contains non-finite number {value}")
        return parsed

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise DuplicateJSONKeyError(f"{label} repeats JSON key {key!r}")
            document[key] = value
        return document

    try:
        text = payload.decode("utf-8", errors="strict")
        document = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except DuplicateJSONKeyError:
        raise
    except CanonicalJSONError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise CanonicalJSONError(f"{label} is not strict UTF-8 JSON") from exc
    if type(document) is not dict:
        raise CanonicalJSONError(f"{label} root must be a JSON object")
    try:
        canonical = canonical_json_bytes(document)
    except CanonicalJSONError as exc:
        raise CanonicalJSONError(f"{label} is not canonical JSON") from exc
    if payload != canonical:
        raise CanonicalJSONError(f"{label} bytes are not canonical JSON")
    frozen = freeze_json(document)
    if type(frozen) is not MappingProxyType:  # pragma: no cover - root asserted above
        raise AssertionError("frozen JSON object did not remain an object")
    return frozen


def require_exact_fields(
    document: Mapping[str, object],
    expected_fields: Collection[str],
    *,
    label: str = "document",
) -> None:
    """Require an object's keys to equal one explicit closed field set."""

    if type(document) not in {dict, MappingProxyType}:
        raise JSONFieldError(f"{label} must be an exact JSON object representation")
    if isinstance(expected_fields, (str, bytes)) or not isinstance(expected_fields, Collection):
        raise JSONFieldError("expected_fields must be a finite collection of strings")
    ordered_fields = tuple(expected_fields)
    if any(type(field_name) is not str for field_name in ordered_fields):
        raise JSONFieldError("expected field names must be exact strings")
    if len(set(ordered_fields)) != len(ordered_fields):
        raise JSONFieldError("expected field names must be unique")
    expected = set(ordered_fields)
    actual = set(document)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise JSONFieldError(
            f"{label} has the wrong closed field set; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _validate_identifier(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise AuthorityValidationError(f"{label} must be a non-empty canonical string")
    return value


def _binding_document(binding: Binding) -> dict[str, object]:
    return {
        "path": binding.path.as_posix(),
        "sha256": binding.sha256,
        "size": binding.size,
    }


def _pending_seal_document(
    *,
    action: str,
    profile: str,
    bindings: tuple[Binding, ...],
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "action": action,
        "authority_type": "pending",
        "bindings": [_binding_document(binding) for binding in bindings],
        "payload": payload,
        "profile": profile,
    }


def _content_seal(document: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PendingAuthority:
    """An immutable, inert request binding an action/profile to captured bytes.

    Construction tokens enforce factory use and content-seal invariants in
    ordinary Python code.  They do not protect against hostile introspection or
    arbitrary memory mutation in this process.
    """

    action: str
    profile: str
    bindings: tuple[Binding, ...] = field(repr=False)
    payload: Mapping[str, object] = field(repr=False)
    content_seal: str
    _issuer: object = field(repr=False)

    def __new__(cls, token: object = None) -> Self:
        if cls is not PendingAuthority or token is not _CONSTRUCTION_TOKEN:
            raise AuthorityConstructionError("PendingAuthority is factory-issued only")
        return object.__new__(cls)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise AuthorityConstructionError("PendingAuthority cannot be subclassed")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ExecutionAuthority:
    """An immutable, inert marker reserved for domain-specific internal issuance.

    This shared module deliberately exposes no public execution-authority
    factory.  Its private token is only an invariant mechanism, not protection
    from hostile Python code.
    """

    pending: PendingAuthority = field(repr=False)
    action: str
    profile: str
    content_seal: str
    _issuer: object = field(repr=False)

    def __new__(cls, token: object = None) -> Self:
        if cls is not ExecutionAuthority or token is not _CONSTRUCTION_TOKEN:
            raise AuthorityConstructionError("ExecutionAuthority is internally issued only")
        return object.__new__(cls)

    def __init_subclass__(cls, **kwargs: object) -> None:
        del kwargs
        raise AuthorityConstructionError("ExecutionAuthority cannot be subclassed")

    @property
    def bindings(self) -> tuple[Binding, ...]:
        return self.pending.bindings

    @property
    def payload(self) -> Mapping[str, object]:
        return self.pending.payload


def issue_pending_authority(
    *,
    action: str,
    profile: str,
    bindings: tuple[Binding, ...],
    payload: Mapping[str, object],
) -> PendingAuthority:
    """Create an inert pending authority after copying and sealing all content."""

    valid_action = _validate_identifier(action, label="action")
    valid_profile = _validate_identifier(profile, label="profile")
    valid_bindings = _validate_snapshot(bindings)
    if type(payload) not in {dict, MappingProxyType}:
        raise AuthorityValidationError("pending payload must be an exact JSON object")
    frozen_payload = freeze_json(payload)
    if type(frozen_payload) is not MappingProxyType:  # pragma: no cover - checked above
        raise AssertionError("pending payload did not remain a JSON object")
    seal = _content_seal(
        _pending_seal_document(
            action=valid_action,
            profile=valid_profile,
            bindings=valid_bindings,
            payload=frozen_payload,
        )
    )
    authority = PendingAuthority.__new__(PendingAuthority, _CONSTRUCTION_TOKEN)
    object.__setattr__(authority, "action", valid_action)
    object.__setattr__(authority, "profile", valid_profile)
    object.__setattr__(authority, "bindings", valid_bindings)
    object.__setattr__(authority, "payload", frozen_payload)
    object.__setattr__(authority, "content_seal", seal)
    object.__setattr__(authority, "_issuer", _PENDING_ISSUER)
    return authority


def _pending_integrity(authority: PendingAuthority) -> None:
    try:
        if authority._issuer is not _PENDING_ISSUER:
            raise AuthorityValidationError("pending authority has an unknown issuer")
        action = _validate_identifier(authority.action, label="pending action")
        profile = _validate_identifier(authority.profile, label="pending profile")
        bindings = _validate_snapshot(authority.bindings)
        if type(authority.payload) is not MappingProxyType:
            raise AuthorityValidationError("pending payload is not factory-frozen")
        expected_seal = _content_seal(
            _pending_seal_document(
                action=action,
                profile=profile,
                bindings=bindings,
                payload=authority.payload,
            )
        )
        _validate_digest(authority.content_seal, label="pending content_seal")
        if not hmac.compare_digest(authority.content_seal, expected_seal):
            raise AuthorityValidationError("pending authority content seal does not match")
    except AuthorityError:
        raise
    except FormalAuthorityError as exc:
        raise AuthorityValidationError("pending authority contains invalid content") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthorityValidationError("pending authority is incomplete") from exc


def require_pending_authority(value: object, *, action: str, profile: str) -> PendingAuthority:
    """Validate exact type, issuer, content seal, and expected action/profile."""

    expected_action = _validate_identifier(action, label="expected action")
    expected_profile = _validate_identifier(profile, label="expected profile")
    if type(value) is not PendingAuthority:
        raise AuthorityValidationError("exact PendingAuthority required")
    _pending_integrity(value)
    if value.action != expected_action or value.profile != expected_profile:
        raise AuthorityMismatchError(
            "pending authority action/profile mismatch: "
            f"expected {expected_action!r}/{expected_profile!r}, "
            f"found {value.action!r}/{value.profile!r}"
        )
    return value


def _execution_seal_document(
    *, pending: PendingAuthority, action: str, profile: str
) -> dict[str, object]:
    return {
        "action": action,
        "authority_type": "execution",
        "pending_content_seal": pending.content_seal,
        "profile": profile,
    }


def _issue_execution_authority(
    pending: PendingAuthority,
    *,
    action: str,
    profile: str,
    _factory_token: object = None,
) -> ExecutionAuthority:
    """Internal hook reserved for a future domain-specific checked factory.

    It is intentionally absent from the package exports.  The token is an API
    invariant only and is not meant to resist hostile in-process code.
    """

    if _factory_token is not _INTERNAL_EXECUTION_FACTORY_TOKEN:
        raise AuthorityConstructionError("execution issuance requires an internal domain factory")
    valid_pending = require_pending_authority(pending, action=action, profile=profile)
    seal = _content_seal(
        _execution_seal_document(pending=valid_pending, action=action, profile=profile)
    )
    authority = ExecutionAuthority.__new__(ExecutionAuthority, _CONSTRUCTION_TOKEN)
    object.__setattr__(authority, "pending", valid_pending)
    object.__setattr__(authority, "action", action)
    object.__setattr__(authority, "profile", profile)
    object.__setattr__(authority, "content_seal", seal)
    object.__setattr__(authority, "_issuer", _EXECUTION_ISSUER)
    return authority


def _execution_integrity(authority: ExecutionAuthority) -> None:
    try:
        if authority._issuer is not _EXECUTION_ISSUER:
            raise AuthorityValidationError("execution authority has an unknown issuer")
        action = _validate_identifier(authority.action, label="execution action")
        profile = _validate_identifier(authority.profile, label="execution profile")
        require_pending_authority(authority.pending, action=action, profile=profile)
        expected_seal = _content_seal(
            _execution_seal_document(
                pending=authority.pending,
                action=action,
                profile=profile,
            )
        )
        _validate_digest(authority.content_seal, label="execution content_seal")
        if not hmac.compare_digest(authority.content_seal, expected_seal):
            raise AuthorityValidationError("execution authority content seal does not match")
    except AuthorityError:
        raise
    except FormalAuthorityError as exc:
        raise AuthorityValidationError("execution authority contains invalid content") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise AuthorityValidationError("execution authority is incomplete") from exc


def require_execution_authority(value: object, *, action: str, profile: str) -> ExecutionAuthority:
    """Validate exact type, issuer, seal, and matching pending action/profile."""

    expected_action = _validate_identifier(action, label="expected action")
    expected_profile = _validate_identifier(profile, label="expected profile")
    if type(value) is not ExecutionAuthority:
        raise AuthorityValidationError("exact ExecutionAuthority required")
    _execution_integrity(value)
    if value.action != expected_action or value.profile != expected_profile:
        raise AuthorityMismatchError(
            "execution authority action/profile mismatch: "
            f"expected {expected_action!r}/{expected_profile!r}, "
            f"found {value.action!r}/{value.profile!r}"
        )
    return value


def revalidate_authority_snapshot(
    root: str | os.PathLike[str],
    value: object,
    *,
    action: str,
    profile: str,
) -> None:
    """Validate an exact authority and freshly recapture every bound path."""

    authority: PendingAuthority | ExecutionAuthority
    if type(value) is PendingAuthority:
        authority = require_pending_authority(value, action=action, profile=profile)
    elif type(value) is ExecutionAuthority:
        authority = require_execution_authority(value, action=action, profile=profile)
    else:
        raise AuthorityValidationError("exact PendingAuthority or ExecutionAuthority required")
    revalidate_snapshot(root, authority.bindings)
