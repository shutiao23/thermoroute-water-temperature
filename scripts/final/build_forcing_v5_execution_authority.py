#!/usr/bin/env python3
"""Build the two-phase forcing-v5 score-execution authority, without outcomes.

The authority is deliberately split across two irreversible, create-only
publication phases:

1. publish a forcing-specific protocol candidate that is sealed as a contract
   but explicitly *does not* authorize scoring;
2. after its SHA256 is pinned in the final runner and that design is committed,
   publish the clean-design record, complete source registry, and terminal seal.

Only the terminal seal authorizes the fixed 12-cell correction.  Neither phase
opens a model shard, score table, prediction file, or formal result directory.
The default invocation is a byte-free dry run; publication is always explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.final.run_forcing_ladder_v5_observed as V5  # noqa: E402


class AuthorityError(RuntimeError):
    """A chronology, binding, or create-only authority check failed closed."""


LOCK_NAME = ".forcing-v5-execution-authority.create.lock"
PROTOCOL_FORMAT = "thermoroute.forcing-v5-observed-protocol-candidate.v1"
TERMINAL_FORMAT = "thermoroute.forcing-v5-observed-execution-seal.v2"


@dataclass(frozen=True)
class PublicationResult:
    phase: str
    files: Mapping[str, Mapping[str, object]]
    execution_authorized: bool


def _relative(path: Path) -> str:
    try:
        return V5._absolute(path).relative_to(V5._absolute(V5.ROOT)).as_posix()
    except ValueError as exc:
        raise AuthorityError(f"authority path escapes repository root: {path}") from exc


def _require_target_path(path: Path, *, label: str) -> Path:
    absolute = V5._absolute(path)
    _relative(absolute)
    if absolute.name in {"", ".", ".."} or "/" in absolute.name or "\0" in absolute.name:
        raise AuthorityError(f"{label} has a non-canonical file name")
    try:
        V5._require_no_symlink_components(absolute.parent, label=f"{label} parent")
    except RuntimeError as exc:
        raise AuthorityError(str(exc)) from exc
    return absolute


def _is_within(path: Path, parent: Path) -> bool:
    try:
        V5._absolute(path).relative_to(V5._absolute(parent))
        return True
    except ValueError:
        return False


def _guard_non_result_read(path: Path, *, label: str) -> None:
    """Keep every byte read outside the formal score/prediction bundle."""

    if _is_within(path, V5.DEFAULT_OUTPUT_DIR):
        raise AuthorityError(f"{label} points into the forbidden formal result directory")
    known_result_roots = (
        V5.FINAL_OUTPUT_ROOT / V5.SHARD_DIRNAME,
        V5.FINAL_OUTPUT_ROOT / "forcing_shards_v5_observed",
    )
    if any(_is_within(path, root) for root in known_result_roots):
        raise AuthorityError(f"{label} points into a score/prediction shard directory")


def _read_allowed(path: Path, *, label: str) -> V5._BoundFile:
    _guard_non_result_read(path, label=label)
    try:
        return V5._read_stable_regular(path, root=V5.ROOT, label=label)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise AuthorityError(str(exc)) from exc


def _require_absent(path: Path, *, label: str) -> None:
    absolute = _require_target_path(path, label=label)
    if os.path.lexists(absolute):
        raise AuthorityError(f"create-only {label} already exists: {absolute}")


def _assert_formal_output_absent() -> None:
    output = V5._absolute(V5.DEFAULT_OUTPUT_DIR)
    _relative(output)
    if os.path.lexists(output):
        raise AuthorityError(
            "formal forcing-v5 result directory already exists; pre-score authority publication "
            "is permanently closed"
        )


def _require_sha256(value: object, *, label: str) -> str:
    try:
        return V5._require_sha256(value, label)
    except ValueError as exc:
        raise AuthorityError(str(exc)) from exc


def _capture_declared(
    paths: Mapping[str, Path],
    expected: Mapping[str, str],
    *,
    category: str,
) -> dict[str, V5._BoundFile]:
    if set(paths) != set(expected):
        raise AuthorityError(f"{category} path and SHA256 role sets differ")
    captured: dict[str, V5._BoundFile] = {}
    for role in sorted(paths):
        expected_sha = _require_sha256(expected[role], label=f"{category} {role} pin")
        bound = _read_allowed(paths[role], label=f"{category} {role}")
        if bound.sha256 != expected_sha:
            raise AuthorityError(
                f"{category} {role} SHA256 mismatch: expected {expected_sha}, "
                f"observed {bound.sha256}"
            )
        captured[role] = bound
    return captured


def _capture_base_authorities() -> dict[str, V5._BoundFile]:
    """Capture only inputs/sources/authorities; never parse outcome material."""

    if V5.EXPECTED_DEFECT_AUTHORITY_SHA256 is None:
        raise AuthorityError("reviewed defect-authority SHA256 pin is unset")
    _require_sha256(
        V5.EXPECTED_DEFECT_AUTHORITY_SHA256,
        label="defect-authority manifest pin",
    )
    captured: dict[str, V5._BoundFile] = {}
    captured.update(
        _capture_declared(
            V5.PINNED_INPUT_PATHS,
            V5.PINNED_INPUT_SHA256,
            category="input authority",
        )
    )
    captured.update(
        _capture_declared(
            V5.PINNED_GOVERNANCE_PATHS,
            V5.PINNED_GOVERNANCE_SHA256,
            category="governance authority",
        )
    )
    captured.update(
        _capture_declared(
            V5.PINNED_DEPENDENCY_PATHS,
            V5.PINNED_DEPENDENCY_SHA256,
            category="source/runtime authority",
        )
    )
    defect_manifest = _read_allowed(
        V5.DEFECT_AUTHORITY_MANIFEST,
        label="defect authority manifest",
    )
    defect_report = _read_allowed(
        V5.DEFECT_AUTHORITY_REPORT,
        label="defect authority report",
    )
    try:
        V5._validate_defect_authority(defect_manifest, defect_report)
    except RuntimeError as exc:
        raise AuthorityError(str(exc)) from exc
    captured["defect_authority_manifest"] = defect_manifest
    captured["defect_authority_report"] = defect_report
    captured["runner"] = _read_allowed(Path(V5.__file__), label="final forcing-v5 runner")
    captured["score_authority_builder"] = _read_allowed(
        Path(__file__),
        label="forcing-v5 execution-authority builder",
    )
    try:
        V5._validate_draft_and_key_authority_semantics(
            captured["protocol"],
            captured["key_authority_manifest"],
        )
        V5._validate_semantic_authority_semantics(captured)
        V5._score_authority_categories(captured)
    except RuntimeError as exc:
        raise AuthorityError(str(exc)) from exc
    return captured


def _bound_payload(path: Path, payload: bytes) -> V5._BoundFile:
    return V5._BoundFile(
        path=V5._absolute(path),
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        stat_signature=(0, 0, stat.S_IFREG, len(payload), 0, 0),
    )


def _protocol_payload(captured: Mapping[str, V5._BoundFile]) -> bytes:
    document = V5._forcing_score_protocol_document(captured)
    payload = V5._canonical_json_bytes(document)
    parsed = yaml.safe_load(payload)
    if parsed != document or document.get("format") != PROTOCOL_FORMAT:
        raise AuthorityError("protocol candidate is not exact canonical YAML/JSON")
    runner_sha = captured["runner"].sha256.encode("ascii")
    builder_sha = captured["score_authority_builder"].sha256.encode("ascii")
    if runner_sha in payload or builder_sha in payload:
        raise AuthorityError("Phase-1 protocol candidate illegally binds runner-bound bytes")
    policy = document.get("result_data_policy")
    if (
        type(policy) is not dict
        or policy.get("model_execution_performed") is not False
        or policy.get("score_or_prediction_artifacts_read") is not False
        or document.get("execution_authorized") is not False
    ):
        raise AuthorityError("Phase-1 protocol candidate is not result-free and inert")
    return payload


def _capture_protocol(
    captured: dict[str, V5._BoundFile],
    *,
    require_runner_pin: bool,
) -> V5._BoundFile:
    protocol = _read_allowed(V5.SEALED_SCORE_PROTOCOL, label="forcing-v5 protocol candidate")
    expected_payload = _protocol_payload(captured)
    if protocol.payload != expected_payload:
        raise AuthorityError("protocol candidate bytes differ from deterministic reconstruction")
    if require_runner_pin:
        expected = _require_sha256(
            V5.EXPECTED_SEALED_SCORE_PROTOCOL_SHA256,
            label="runner protocol-candidate pin",
        )
        if protocol.sha256 != expected:
            raise AuthorityError("protocol candidate differs from the final runner pin")
    captured["sealed_score_protocol"] = protocol
    return protocol


def _parse_canonical_json(bound: V5._BoundFile, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(bound.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"{label} is not valid JSON") from exc
    if type(document) is not dict or bound.payload != V5._canonical_json_bytes(document):
        raise AuthorityError(f"{label} is not exact canonical JSON")
    return document


def _terminal_payloads(
    captured: Mapping[str, V5._BoundFile],
    git_design: Mapping[str, object],
) -> tuple[bytes, bytes, bytes]:
    clean_document = V5._score_clean_design_document(captured, git_design)
    clean_payload = V5._canonical_json_bytes(clean_document)
    clean_bound = _bound_payload(V5.SCORE_CLEAN_DESIGN_COMMIT, clean_payload)
    source_document = V5._score_source_registry_document(
        captured,
        clean_bound,
        git_design,
    )
    source_payload = V5._canonical_json_bytes(source_document)
    source_bound = _bound_payload(V5.SCORE_SOURCE_REGISTRY, source_payload)
    seal_document = V5._score_terminal_seal_document(
        captured,
        clean_bound,
        source_bound,
        git_design,
    )
    seal_payload = V5._canonical_json_bytes(seal_document)
    for payload, expected_format, label in (
        (
            clean_payload,
            "thermoroute.forcing-v5-observed-clean-design-commit.v2",
            "clean-design record",
        ),
        (
            source_payload,
            "thermoroute.forcing-v5-observed-source-registry.v2",
            "source registry",
        ),
        (seal_payload, TERMINAL_FORMAT, "terminal seal"),
    ):
        parsed = json.loads(payload)
        if type(parsed) is not dict or parsed.get("format") != expected_format:
            raise AuthorityError(f"{label} construction failed")
    return clean_payload, source_payload, seal_payload


def _verify_terminal_chain(
    captured: dict[str, V5._BoundFile],
    clean: V5._BoundFile,
    source: V5._BoundFile,
    seal: V5._BoundFile,
) -> None:
    _parse_canonical_json(clean, label="clean-design record")
    _parse_canonical_json(source, label="source registry")
    _parse_canonical_json(seal, label="terminal seal")
    captured["score_clean_design_commit"] = clean
    captured["score_source_registry"] = source
    captured["sealed_score_protocol_seal"] = seal
    try:
        V5._validate_score_execution_authority(
            captured["sealed_score_protocol"],
            seal,
            clean,
            source,
            captured,
        )
    except RuntimeError as exc:
        raise AuthorityError(str(exc)) from exc


def _anchored_parent(path: Path, *, label: str) -> tuple[int, tuple[int, int], str]:
    absolute = _require_target_path(path, label=label)
    parent = absolute.parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise AuthorityError(f"cannot anchor {label} parent: {parent}") from exc
    status = os.fstat(descriptor)
    if not stat.S_ISDIR(status.st_mode):
        os.close(descriptor)
        raise AuthorityError(f"{label} parent is not a directory")
    identity = (status.st_dev, status.st_ino)
    try:
        V5._verify_anchored_directory_path(parent, descriptor, identity, label=label)
    except RuntimeError as exc:
        os.close(descriptor)
        raise AuthorityError(str(exc)) from exc
    return descriptor, identity, absolute.name


def _entry_at(descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise AuthorityError("short write while staging authority artifact")
        offset += written


def _read_at(descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    opened = os.open(name, flags, dir_fd=descriptor)
    try:
        before = os.fstat(opened)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorityError("staged authority artifact is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(opened, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(opened)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AuthorityError("staged authority artifact changed while being verified")
        return b"".join(chunks)
    finally:
        os.close(opened)


def _unlink_owned(descriptor: int, name: str, identity: tuple[int, int]) -> None:
    current = _entry_at(descriptor, name)
    if current is None:
        return
    if (current.st_dev, current.st_ino) != identity:
        raise AuthorityError("refusing to remove a replaced authority publication entry")
    os.unlink(name, dir_fd=descriptor)


def _publish_create_only(
    files: Sequence[tuple[Path, bytes]],
    *,
    precommit: Callable[[], None],
) -> dict[str, dict[str, object]]:
    """Stage, verify, and link exact files create-only; terminal must be last."""

    if not files:
        raise AuthorityError("create-only publication has no files")
    paths = [V5._absolute(path) for path, _payload in files]
    if len(set(paths)) != len(paths):
        raise AuthorityError("create-only publication contains duplicate targets")
    root_descriptor, root_identity, _root_name = _anchored_parent(
        V5.ROOT / LOCK_NAME,
        label="authority publication lock",
    )
    lock_identity: tuple[int, int] | None = None
    parents: list[tuple[int, tuple[int, int], str, Path, bytes]] = []
    staged: list[tuple[int, str, tuple[int, int]]] = []
    published: list[tuple[int, str, tuple[int, int]]] = []
    try:
        lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        lock_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            lock_fd = os.open(LOCK_NAME, lock_flags, 0o600, dir_fd=root_descriptor)
        except FileExistsError as exc:
            raise AuthorityError("another create-only authority publication is active") from exc
        try:
            _write_all(lock_fd, b"forcing-v5-execution-authority\n")
            os.fsync(lock_fd)
            lock_status = os.fstat(lock_fd)
            lock_identity = (lock_status.st_dev, lock_status.st_ino)
        finally:
            os.close(lock_fd)
        os.fsync(root_descriptor)

        for index, (path, payload) in enumerate(files):
            descriptor, identity, name = _anchored_parent(path, label=f"authority target {index}")
            parents.append((descriptor, identity, name, path.parent, payload))
            if _entry_at(descriptor, name) is not None:
                raise AuthorityError(f"create-only authority collision: {path}")

        for index, (descriptor, _identity, name, _parent, payload) in enumerate(parents):
            stage_name = f".{name}.staging.{secrets.token_hex(16)}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            stage_fd = os.open(stage_name, flags, 0o444, dir_fd=descriptor)
            try:
                _write_all(stage_fd, payload)
                os.fsync(stage_fd)
                stage_status = os.fstat(stage_fd)
                stage_identity = (stage_status.st_dev, stage_status.st_ino)
            finally:
                os.close(stage_fd)
            staged.append((descriptor, stage_name, stage_identity))
            if _read_at(descriptor, stage_name) != payload:
                raise AuthorityError(f"staged authority artifact {index} differs from payload")
            os.fsync(descriptor)

        precommit()
        V5._verify_anchored_directory_path(
            V5._absolute(V5.ROOT),
            root_descriptor,
            root_identity,
            label="authority root precommit",
        )
        for descriptor, identity, name, parent, _payload in parents:
            V5._verify_anchored_directory_path(
                V5._absolute(parent),
                descriptor,
                identity,
                label=f"authority parent precommit {name}",
            )
            if _entry_at(descriptor, name) is not None:
                raise AuthorityError(
                    f"create-only authority target appeared during staging: {name}"
                )

        bindings: dict[str, dict[str, object]] = {}
        for (descriptor, identity, name, parent, payload), (
            stage_descriptor,
            stage_name,
            stage_identity,
        ) in zip(parents, staged, strict=True):
            if descriptor != stage_descriptor:
                raise AuthorityError("authority staging parent identity changed")
            V5._verify_anchored_directory_path(
                V5._absolute(V5.ROOT),
                root_descriptor,
                root_identity,
                label="authority root during commit",
            )
            V5._verify_anchored_directory_path(
                V5._absolute(parent),
                descriptor,
                identity,
                label=f"authority parent during commit {name}",
            )
            try:
                os.link(
                    stage_name,
                    name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise AuthorityError(
                    f"create-only authority collision during commit: {name}"
                ) from exc
            target = _entry_at(descriptor, name)
            if target is None or (target.st_dev, target.st_ino) != stage_identity:
                raise AuthorityError("authority target does not name the staged inode")
            published.append((descriptor, name, stage_identity))
            if _read_at(descriptor, name) != payload:
                raise AuthorityError("published authority bytes differ before terminal commit")
            V5._verify_anchored_directory_path(
                V5._absolute(parent),
                descriptor,
                identity,
                label=f"authority parent after commit {name}",
            )
            os.unlink(stage_name, dir_fd=descriptor)
            os.fsync(descriptor)
            path = Path(parent) / name
            bindings[_relative(path)] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        return bindings
    except Exception:
        cleanup_error: Exception | None = None
        for descriptor, name, identity in reversed(published):
            try:
                _unlink_owned(descriptor, name, identity)
                os.fsync(descriptor)
            except (AuthorityError, OSError) as exc:  # pragma: no cover - inode replacement
                cleanup_error = cleanup_error or exc
        for descriptor, name, identity in reversed(staged):
            try:
                _unlink_owned(descriptor, name, identity)
                os.fsync(descriptor)
            except (AuthorityError, OSError) as exc:  # pragma: no cover - inode replacement
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            raise AuthorityError("authority publication failed and safe rollback was incomplete")
        raise
    finally:
        for descriptor, _identity, _name, _parent, _payload in parents:
            os.close(descriptor)
        try:
            if lock_identity is not None:
                _unlink_owned(root_descriptor, LOCK_NAME, lock_identity)
                os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)


def _same_snapshot(
    first: Mapping[str, V5._BoundFile],
    second: Mapping[str, V5._BoundFile],
) -> bool:
    if set(first) != set(second):
        return False
    return all(
        first[role].payload == second[role].payload
        and first[role].sha256 == second[role].sha256
        and first[role].stat_signature == second[role].stat_signature
        for role in first
    )


def publish_protocol_candidate() -> PublicationResult:
    """Phase 1: create the inert, result-free canonical protocol candidate."""

    _assert_formal_output_absent()
    _require_absent(V5.SEALED_SCORE_PROTOCOL, label="protocol candidate")
    _require_absent(V5.SCORE_CLEAN_DESIGN_COMMIT, label="clean-design record")
    _require_absent(V5.SCORE_SOURCE_REGISTRY, label="source registry")
    _require_absent(V5.SEALED_SCORE_PROTOCOL_SEAL, label="terminal seal")
    if V5.EXPECTED_SEALED_SCORE_PROTOCOL_SHA256 is not None:
        raise AuthorityError(
            "Phase 1 is closed because the runner already declares a protocol pin; "
            "a candidate must be created before that pin"
        )
    first = _capture_base_authorities()
    payload = _protocol_payload(first)

    def precommit() -> None:
        _assert_formal_output_absent()
        second = _capture_base_authorities()
        if not _same_snapshot(first, second) or _protocol_payload(second) != payload:
            raise AuthorityError("Phase-1 authorities changed before create-only commit")

    files = _publish_create_only(
        ((V5.SEALED_SCORE_PROTOCOL, payload),),
        precommit=precommit,
    )
    published = _read_allowed(V5.SEALED_SCORE_PROTOCOL, label="published protocol candidate")
    if published.payload != payload:
        raise AuthorityError("published protocol candidate changed after commit")
    return PublicationResult(
        phase="PROTOCOL_CANDIDATE_PUBLISHED_AWAITING_RUNNER_PIN_AND_GIT_COMMIT",
        files=files,
        execution_authorized=False,
    )


def publish_terminal_authority() -> PublicationResult:
    """Phase 2: publish the terminal chain after the exact design is committed."""

    _assert_formal_output_absent()
    _require_absent(V5.SCORE_CLEAN_DESIGN_COMMIT, label="clean-design record")
    _require_absent(V5.SCORE_SOURCE_REGISTRY, label="source registry")
    _require_absent(V5.SEALED_SCORE_PROTOCOL_SEAL, label="terminal seal")
    first = _capture_base_authorities()
    _capture_protocol(first, require_runner_pin=True)
    try:
        git_design = V5._git_design_authority_record(first)
    except RuntimeError as exc:
        raise AuthorityError(str(exc)) from exc
    clean_payload, source_payload, seal_payload = _terminal_payloads(first, git_design)

    def precommit() -> None:
        _assert_formal_output_absent()
        second = _capture_base_authorities()
        _capture_protocol(second, require_runner_pin=True)
        if not _same_snapshot(first, second):
            raise AuthorityError("Phase-2 authorities changed before terminal commit")
        try:
            current_git = V5._git_design_authority_record(second)
        except RuntimeError as exc:
            raise AuthorityError(str(exc)) from exc
        if current_git != git_design:
            raise AuthorityError("clean-design Git commit/tree changed before terminal commit")
        if _terminal_payloads(second, current_git) != (
            clean_payload,
            source_payload,
            seal_payload,
        ):
            raise AuthorityError("terminal authority reconstruction changed before commit")

    files = _publish_create_only(
        (
            (V5.SCORE_CLEAN_DESIGN_COMMIT, clean_payload),
            (V5.SCORE_SOURCE_REGISTRY, source_payload),
            # The only affirmative authority is linked last.
            (V5.SEALED_SCORE_PROTOCOL_SEAL, seal_payload),
        ),
        precommit=precommit,
    )
    clean = _read_allowed(V5.SCORE_CLEAN_DESIGN_COMMIT, label="published clean-design record")
    source = _read_allowed(V5.SCORE_SOURCE_REGISTRY, label="published source registry")
    seal = _read_allowed(V5.SEALED_SCORE_PROTOCOL_SEAL, label="published terminal seal")
    _verify_terminal_chain(first, clean, source, seal)
    return PublicationResult(
        phase="TERMINAL_SCORE_EXECUTION_AUTHORITY_PUBLISHED",
        files=files,
        execution_authorized=True,
    )


def verify_authority() -> dict[str, object]:
    """Read only authority/input/source bytes and verify the published phase."""

    captured = _capture_base_authorities()
    protocol = _capture_protocol(
        captured,
        require_runner_pin=V5.EXPECTED_SEALED_SCORE_PROTOCOL_SHA256 is not None,
    )
    terminal_paths = (
        V5.SCORE_CLEAN_DESIGN_COMMIT,
        V5.SCORE_SOURCE_REGISTRY,
        V5.SEALED_SCORE_PROTOCOL_SEAL,
    )
    existence = tuple(os.path.lexists(path) for path in terminal_paths)
    if not any(existence):
        return {
            "verified": True,
            "phase": "PROTOCOL_CANDIDATE_ONLY_NOT_SCORE_AUTHORITY",
            "execution_authorized": False,
            "protocol": V5._content_binding(protocol),
            "score_or_prediction_artifacts_read": False,
        }
    if not all(existence):
        raise AuthorityError("terminal authority publication is partial and therefore inert")
    if V5.EXPECTED_SEALED_SCORE_PROTOCOL_SHA256 is None:
        raise AuthorityError("terminal files exist but the final runner protocol pin is unset")
    clean = _read_allowed(V5.SCORE_CLEAN_DESIGN_COMMIT, label="clean-design record")
    source = _read_allowed(V5.SCORE_SOURCE_REGISTRY, label="source registry")
    seal = _read_allowed(V5.SEALED_SCORE_PROTOCOL_SEAL, label="terminal seal")
    _verify_terminal_chain(captured, clean, source, seal)
    return {
        "verified": True,
        "phase": "TERMINAL_SCORE_EXECUTION_AUTHORITY",
        "execution_authorized": True,
        "protocol": V5._content_binding(protocol),
        "clean_design_commit": V5._content_binding(clean),
        "source_registry": V5._content_binding(source),
        "terminal_seal": V5._content_binding(seal),
        "score_or_prediction_artifacts_read": False,
    }


def plan_record() -> dict[str, object]:
    """Return a byte-free plan.  Do not inspect filesystem or Git state here."""

    return {
        "format": "thermoroute.forcing-v5-observed-execution-authority-plan.v1",
        "execution_performed": False,
        "model_execution_performed": False,
        "score_or_prediction_artifacts_read": False,
        "scope": V5._score_execution_scope_record(),
        "phase_1": {
            "action": "CREATE_PROTOCOL_CANDIDATE",
            "path": _relative(V5.SEALED_SCORE_PROTOCOL),
            "execution_authorized_after_phase": False,
            "create_only": True,
        },
        "handoff": {
            "pin_protocol_sha256_in_final_runner": True,
            "commit_final_runner_builder_protocol_and_design_sources": True,
            "require_exact_git_commit_and_tree": True,
        },
        "phase_2": {
            "action": "CREATE_TERMINAL_AUTHORITY_CHAIN",
            "ordered_paths": [
                _relative(V5.SCORE_CLEAN_DESIGN_COMMIT),
                _relative(V5.SCORE_SOURCE_REGISTRY),
                _relative(V5.SEALED_SCORE_PROTOCOL_SEAL),
            ],
            "terminal_seal_linked_last": True,
            "create_only": True,
        },
        "runner_pins": {
            "defect_authority_sha256": V5.EXPECTED_DEFECT_AUTHORITY_SHA256,
            "protocol_candidate_sha256": V5.EXPECTED_SEALED_SCORE_PROTOCOL_SHA256,
        },
        "formal_output": _relative(V5.DEFAULT_OUTPUT_DIR),
        "formal_output_must_be_absent_during_publication": True,
        "dry_run_reads_files_or_git": False,
    }


def _result_json(result: PublicationResult) -> dict[str, object]:
    return {
        "phase": result.phase,
        "execution_authorized": result.execution_authorized,
        "files": dict(result.files),
        "model_execution_performed": False,
        "score_or_prediction_artifacts_read": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print the two-phase plan without reading files or Git (default)",
    )
    mode.add_argument(
        "--create-protocol-candidate",
        "--phase-1",
        dest="create_protocol_candidate",
        action="store_true",
        help="create the inert Phase-1 protocol candidate at its canonical path",
    )
    mode.add_argument(
        "--publish-terminal-authority",
        "--phase-2",
        dest="publish_terminal_authority",
        action="store_true",
        help="after runner pin + clean Git commit, create the terminal authority chain",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="verify whichever published authority phase exists; write nothing",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if bool(getattr(args, "create_protocol_candidate", False)):
        output = _result_json(publish_protocol_candidate())
    elif bool(getattr(args, "publish_terminal_authority", False)):
        output = _result_json(publish_terminal_authority())
    elif bool(getattr(args, "verify", False)):
        output = verify_authority()
    else:
        output = plan_record()
    print(json.dumps(output, indent=1, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (AuthorityError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    sys.exit(main())
