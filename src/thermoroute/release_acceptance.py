"""Same-host, fresh-process acceptance for the local Route-A evidence archive.

This module deliberately records a narrow engineering observation.  It copies
the already hashed interpreter, release verifier, archive, and sidecar into a
private temporary tree, launches those sealed bytes in a distinct
``python -I -B`` process group from a fresh working directory and allowlisted
environment, then records the invocation and one OS resource observation in a
create-only receipt.  Receipt validation performs a new fixed replay from the
currently bound archive; it does not authenticate the historical execution or
its resource numbers.  It does not execute the Route-A opening and it does not
claim a fresh machine, container, virtual environment, cold cache, network or
file-read sandbox, energy measurement, multi-hardware replay, or deployment
latency.

The only accepted profile is ``PREOPEN_NOT_COMPLETE``.  Consequently, a PASS is
named release-*mechanics* acceptance rather than model-suite readiness or a
scientific result.  The archive verifier remains responsible for rejecting
labels and confirmation namespaces.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn


RECEIPT_FORMAT = "thermoroute.same-host-release-mechanics-acceptance.v2"
RECEIPT_STATUS = "PASS_SAME_HOST_FRESH_PROCESS_RELEASE_MECHANICS"
RELEASE_PROFILE = "PREOPEN_NOT_COMPLETE"
LOCAL_DISTRIBUTION = "LOCAL_EVIDENCE_ONLY"
EVIDENCE_SCOPE = (
    "SAME_HOST_RELEASE_MECHANICS_ACCEPTANCE_ONLY_NOT_MODEL_READINESS_NOT_SCIENTIFIC_RESULT"
)

CORE_RELATIVE = "src/thermoroute/release_acceptance.py"
RUNNER_RELATIVE = "scripts/30_verify_release_fresh_process.py"
VERIFIER_RELATIVE = "scripts/verify_release.py"
RECEIPT_RELATIVE = "outputs/prelabel/route_a_release_mechanics_acceptance_v2.json"

MEASUREMENT_BACKEND = "perf_counter_ns_plus_exact_wait4_process_tree_rusage_v2"
MEASUREMENT_SCOPE = (
    "ONE_FIXED_VERIFIER_PROCESS_TREE_ON_THIS_HOST;_CPU_AND_RSS_ARE_THE_OS_"
    "WAIT4_SUMMARY_FOR_THE_VERIFIER_AND_ALL_OF_ITS_DESCENDANTS;_PEAK_RSS_IS_"
    "NOT_AN_ADDITIVE_OR_SIMULTANEOUS_PROCESS_TREE_MEMORY_PEAK"
)
NORMALISED_COMMAND = [
    "<sealed-copy-of-bound-python-executable>",
    "-I",
    "-B",
    "<sealed-copy-of-git-bound-release-verifier>",
    "<sealed-copy-of-bound-archive>",
    "--distribution",
    LOCAL_DISTRIBUTION,
]
PROFILE_EVIDENCE_LINE = "LOCAL EVIDENCE OK [DO NOT DISTRIBUTE; PREOPEN_NOT_COMPLETE]: <archive>"

_DIGEST = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_MANIFEST_TRANSCRIPT_LINE = re.compile(
    r"manifest OK: [0-9]+ artifacts, source [0-9a-f]{12}, DAG [0-9]+ nodes"
)
_MAX_CAPTURE_BYTES = 8 * 1024 * 1024
_VERIFIER_DEADLINE_SECONDS = 60 * 60
_PROCESS_POLL_SECONDS = 0.01
_PROCESS_TERMINATION_GRACE_SECONDS = 2.0

_CHILD_ENVIRONMENT_STABLE = {
    "PATH": os.defpath,
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
CHILD_ENVIRONMENT_CONTRACT = {
    **_CHILD_ENVIRONMENT_STABLE,
    "TMPDIR": "<fresh-temporary-root>",
}

ISOLATION_LIMITATIONS = {
    "fresh_machine": False,
    "container": False,
    "fresh_virtual_environment": False,
    "dependency_reinstallation_from_hashed_lock": False,
    "cold_cache_enforced": False,
    "network_isolation_enforced": False,
    "filesystem_read_isolation_enforced": False,
    "descendant_python_isolation_flags_enforced": False,
    "descendant_new_session_escape_prevented": False,
    "energy_measured": False,
    "multi_hardware_replay": False,
    "deployment_latency_claim_allowed": False,
}

AUTHENTICATION_LIMITATIONS = {
    "external_authentication": False,
    "historical_execution_independently_proven": False,
    "resource_observation_independently_attested": False,
    "receipt_owner_tamper_resistance": False,
    "same_account_concurrent_adversary_resistance": False,
    "self_hash_role": "INTERNAL_CONSISTENCY_AND_ACCIDENTAL_CORRUPTION_DETECTION_ONLY",
}

LABEL_SAFETY = {
    "runner_invoked_opening_entrypoint": False,
    "accepted_profile_declares_labels_included": False,
    "confirmation_namespace_allowed_by_accepted_profile": False,
    "post_2020_outcome_read_tracing_enforced": False,
    "outcome_read_absence_claim_allowed": False,
}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "format",
        "status",
        "profile",
        "distribution",
        "evidence_scope",
        "archive",
        "source",
        "runtime",
        "execution",
        "resource_observation",
        "isolation_limitations",
        "authentication_limitations",
        "label_safety",
        "receipt_self_sha256",
    }
)
_BINDING_FIELDS = frozenset({"path", "sha256", "bytes"})
_ARCHIVE_FIELDS = frozenset({"path", "sha256", "bytes", "sidecar"})
_SOURCE_FIELDS = frozenset(
    {
        "git_commit",
        "git_tree",
        "git_clean_before_acceptance",
        "core",
        "runner",
        "verifier",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "python_implementation",
        "python_version",
        "python_executable",
        "platform_system",
        "platform_release",
        "machine",
    }
)
_PYTHON_FIELDS = frozenset({"invoked_path", "realpath", "sha256", "bytes"})
_EXECUTION_FIELDS = frozenset(
    {
        "command",
        "environment",
        "controller_pid",
        "verifier_pid",
        "python_isolated",
        "bytecode_disabled",
        "fresh_process",
        "fresh_working_directory",
        "fresh_temporary_extraction_root",
        "fresh_cwd_empty_before",
        "fresh_cwd_empty_after",
        "temporary_root_removed_after_exit",
        "sealed_archive_argument",
        "process_group_cleanup_enforced",
        "returncode",
        "stdout_sha256",
        "stdout_bytes",
        "stdout_transcript",
        "stderr_sha256",
        "stderr_bytes",
        "stderr_transcript",
        "profile_evidence_line",
    }
)
_RESOURCE_FIELDS = frozenset(
    {
        "wall_time_ns",
        "user_cpu_time_ns",
        "system_cpu_time_ns",
        "total_cpu_time_ns",
        "peak_rss_raw",
        "peak_rss_raw_unit",
        "peak_rss_bytes",
        "measurement_backend",
        "measurement_scope",
        "repetitions",
    }
)


class ReleaseAcceptanceError(RuntimeError):
    """The same-host release-mechanics acceptance contract failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _duplicate_rejecting_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReleaseAcceptanceError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> NoReturn:
    raise ReleaseAcceptanceError(f"non-finite JSON constant: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except ReleaseAcceptanceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAcceptanceError(f"cannot parse {label}") from exc
    if not isinstance(value, dict):
        raise ReleaseAcceptanceError(f"{label} must be a JSON object")
    return value


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseAcceptanceError(f"cannot open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ReleaseAcceptanceError(f"{label} must be one non-linked regular file: {path}")
        if max_bytes is not None and metadata.st_size > max_bytes:
            raise ReleaseAcceptanceError(f"{label} exceeds its byte limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ReleaseAcceptanceError(f"{label} exceeds its byte limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _payload_binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _normalise_relative(value: str, *, label: str) -> str:
    if (
        not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReleaseAcceptanceError(f"{label} path is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ReleaseAcceptanceError(f"{label} path is not canonical and relative")
    return value


def _inside(root: Path, relative: str, *, label: str) -> Path:
    relative = _normalise_relative(relative, label=label)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ReleaseAcceptanceError(f"{label} path traverses a symbolic link")
    resolved_parent = candidate.parent.resolve()
    if resolved_parent != root and root not in resolved_parent.parents:
        raise ReleaseAcceptanceError(f"{label} path escapes repository")
    return candidate


def _relative_existing(root: Path, path: str | Path, *, label: str) -> tuple[str, Path]:
    supplied = Path(path)
    absolute = supplied if supplied.is_absolute() else root / supplied
    absolute = Path(os.path.abspath(absolute))
    try:
        relative = absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseAcceptanceError(f"{label} is outside repository") from exc
    expected = _inside(root, relative, label=label)
    if absolute != expected:
        raise ReleaseAcceptanceError(f"{label} path uses a noncanonical alias")
    return relative, expected


def _binding(root: Path, path: Path, *, label: str) -> dict[str, object]:
    relative, canonical = _relative_existing(root, path, label=label)
    payload = _read_regular_bytes(canonical, label=label)
    return _payload_binding(relative, payload)


def _interpreter_material() -> tuple[dict[str, object], bytes]:
    invoked = Path(sys.executable)
    if not invoked.is_absolute() or not invoked.exists():
        raise ReleaseAcceptanceError("Python executable path is not absolute or absent")
    realpath = invoked.resolve()
    payload = _read_regular_bytes(realpath, label="Python executable")
    return (
        {
            "invoked_path": str(invoked),
            "realpath": str(realpath),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        payload,
    )


def _interpreter_binding() -> dict[str, object]:
    return _interpreter_material()[0]


def _git_environment() -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git(root: Path, *arguments: str, check: bool = True) -> bytes:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise ReleaseAcceptanceError("cannot find Git in the fixed system PATH")
    result = subprocess.run(
        [
            executable,
            "--no-replace-objects",
            "-c",
            "core.useReplaceRefs=false",
            "-C",
            str(root),
            *arguments,
        ],
        env=_git_environment(),
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseAcceptanceError(
            f"Git command failed ({' '.join(arguments)}): {detail[-2000:]}"
        )
    return result.stdout


def _assert_exact_git_root(root: Path) -> None:
    top = _git(root, "rev-parse", "--show-toplevel").decode().strip()
    if Path(top).resolve() != root:
        raise ReleaseAcceptanceError("acceptance requires the exact Git worktree root")
    shallow = _git(root, "rev-parse", "--is-shallow-repository").decode().strip()
    if shallow != "false":
        raise ReleaseAcceptanceError("shallow or indeterminate Git history is prohibited")
    common = Path(_git(root, "rev-parse", "--git-common-dir").decode().strip())
    if not common.is_absolute():
        common = root / common
    common = common.resolve()
    objects = (common / "objects").resolve()
    if not objects.is_dir() or (objects / "info" / "alternates").exists():
        raise ReleaseAcceptanceError("Git object store is absent or uses alternates")
    for forbidden in (common / "shallow", common / "info" / "grafts"):
        if forbidden.exists():
            raise ReleaseAcceptanceError("Git history overlay is prohibited")
    replace = _git(root, "for-each-ref", "--format=%(refname)", "refs/replace").strip()
    if replace:
        raise ReleaseAcceptanceError("Git replacement refs are prohibited")


def _source_material(
    root: Path,
    *,
    require_clean: bool,
) -> tuple[dict[str, object], dict[str, bytes]]:
    _assert_exact_git_root(root)
    if require_clean:
        dirt = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if dirt:
            lines = dirt.decode("utf-8", errors="replace").splitlines()
            raise ReleaseAcceptanceError(
                f"release acceptance requires a clean Git worktree: {lines[:10]}"
            )
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    if _GIT_OBJECT.fullmatch(commit) is None or _GIT_OBJECT.fullmatch(tree) is None:
        raise ReleaseAcceptanceError("Git commit/tree identity is malformed")
    bindings: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    for label, relative in (
        ("core", CORE_RELATIVE),
        ("runner", RUNNER_RELATIVE),
        ("verifier", VERIFIER_RELATIVE),
    ):
        path = _inside(root, relative, label=label)
        current = _read_regular_bytes(path, label=label)
        committed = _git(root, "show", f"{commit}:{relative}")
        if committed != current:
            raise ReleaseAcceptanceError(f"{label} bytes differ from the acceptance Git commit")
        bindings[label] = _payload_binding(relative, current)
        payloads[label] = current
    return (
        {
            "git_commit": commit,
            "git_tree": tree,
            "git_clean_before_acceptance": True,
            **bindings,
        },
        payloads,
    )


def _source_identity(
    root: Path,
    *,
    require_clean: bool,
) -> dict[str, object]:
    return _source_material(root, require_clean=require_clean)[0]


def _validate_source_identity(
    root: Path,
    source: Mapping[str, Any],
) -> dict[str, bytes]:
    _require_keys(source, _SOURCE_FIELDS, label="source")
    commit, tree = source.get("git_commit"), source.get("git_tree")
    if (
        not isinstance(commit, str)
        or _GIT_OBJECT.fullmatch(commit) is None
        or not isinstance(tree, str)
        or _GIT_OBJECT.fullmatch(tree) is None
        or source.get("git_clean_before_acceptance") is not True
    ):
        raise ReleaseAcceptanceError("receipt Git source identity is malformed")
    _assert_exact_git_root(root)
    actual_tree = _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if actual_tree != tree:
        raise ReleaseAcceptanceError("receipt Git tree differs from its commit")
    # Use an explicit history enumeration because the small Git wrapper returns
    # captured stdout rather than a CompletedProcess return code.
    head_history = set(_git(root, "rev-list", "HEAD").decode("ascii", errors="strict").splitlines())
    if commit not in head_history:
        raise ReleaseAcceptanceError("receipt source commit is not an ancestor of HEAD")
    payloads: dict[str, bytes] = {}
    for label, relative in (
        ("core", CORE_RELATIVE),
        ("runner", RUNNER_RELATIVE),
        ("verifier", VERIFIER_RELATIVE),
    ):
        binding, current = _validate_repository_binding(
            root,
            source.get(label),
            expected_path=relative,
            label=label,
        )
        committed = _git(root, "show", f"{commit}:{relative}")
        if hashlib.sha256(committed).hexdigest() != binding.get("sha256") or len(
            committed
        ) != binding.get("bytes"):
            raise ReleaseAcceptanceError(f"receipt {label} differs from its source Git blob")
        payloads[label] = current
    return payloads


def _child_environment(temporary_root: Path) -> dict[str, str]:
    environment = {
        **_CHILD_ENVIRONMENT_STABLE,
        "TMPDIR": str(temporary_root),
    }
    forbidden = [
        key
        for key in environment
        if key.startswith(("PYTHON", "GIT_", "DYLD")) or key == "LD_PRELOAD"
    ]
    if forbidden:
        raise ReleaseAcceptanceError(
            f"fixed verifier environment contains injectable fields: {forbidden}"
        )
    return environment


def _seconds_to_ns(value: float, *, label: str) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ReleaseAcceptanceError(f"non-finite or negative {label}")
    return int(round(value * 1_000_000_000))


def _rss_contract(raw: int) -> tuple[str, int]:
    if type(raw) is not int or raw < 0:
        raise ReleaseAcceptanceError("peak RSS observation is malformed")
    system = platform.system()
    if system == "Darwin":
        return "bytes", raw
    if system == "Linux":
        return "kibibytes", raw * 1024
    raise ReleaseAcceptanceError(f"peak RSS normalization is unsupported on {system!r}")


def _wait4(pid: int, options: int = 0) -> tuple[int, int, Any]:
    while True:
        try:
            waited, status, usage = os.wait4(pid, options)
        except InterruptedError:
            continue
        if waited not in {0, pid}:
            raise ReleaseAcceptanceError("wait4 returned another process identity")
        return waited, status, usage


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise ReleaseAcceptanceError("sealed input copy made no write progress")
        view = view[written:]


def _write_sealed_payload(
    directory: Path,
    *,
    name: str,
    payload: bytes,
    mode: int,
    label: str,
) -> Path:
    if not name or PurePosixPath(name).name != name or "/" in name or "\\" in name:
        raise ReleaseAcceptanceError(f"{label} sealed-copy name is unsafe")
    directory_descriptor = os.open(
        directory,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor = os.open(name, flags, mode, dir_fd=directory_descriptor)
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    sealed = directory / name
    if _read_regular_bytes(sealed, label=f"sealed {label}") != payload:
        raise ReleaseAcceptanceError(f"sealed {label} bytes differ from their verified source FD")
    return sealed


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError as exc:
        raise ReleaseAcceptanceError(
            "cannot inspect verifier descendant process group"
        ) from exc
    return True


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise ReleaseAcceptanceError(
            "cannot signal verifier descendant process group"
        ) from exc


def _signal_direct_child(pid: int, signal_number: int) -> None:
    try:
        os.kill(pid, signal_number)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise ReleaseAcceptanceError("cannot signal verifier direct child") from exc


def _terminate_and_reap(
    pid: int,
    *,
    already_reaped: tuple[int, Any] | None,
) -> tuple[int, Any]:
    status_usage = already_reaped
    group_cleanup_error: ReleaseAcceptanceError | None = None
    try:
        _signal_process_group(pid, signal.SIGTERM)
    except ReleaseAcceptanceError as exc:
        # A process-group permission failure must remain fail-closed, but it
        # must not prevent us from killing and reaping the direct child whose
        # PID is still unambiguous while it remains our unreaped child.
        group_cleanup_error = exc
    grace_deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    while status_usage is None and time.monotonic() < grace_deadline:
        waited, status, usage = _wait4(pid, os.WNOHANG)
        if waited == pid:
            status_usage = (status, usage)
            break
        time.sleep(_PROCESS_POLL_SECONDS)

    if group_cleanup_error is None:
        try:
            if _process_group_exists(pid):
                _signal_process_group(pid, signal.SIGKILL)
        except ReleaseAcceptanceError as exc:
            group_cleanup_error = exc

    if status_usage is None:
        direct_cleanup_error: ReleaseAcceptanceError | None = None
        try:
            _signal_direct_child(pid, signal.SIGKILL)
        except ReleaseAcceptanceError as exc:
            direct_cleanup_error = exc
        reap_deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
        while status_usage is None and time.monotonic() < reap_deadline:
            waited, status, usage = _wait4(pid, os.WNOHANG)
            if waited == pid:
                status_usage = (status, usage)
                break
            time.sleep(_PROCESS_POLL_SECONDS)
        if status_usage is None:
            error = ReleaseAcceptanceError(
                "verifier direct child survived forced cleanup"
            )
            if direct_cleanup_error is not None:
                raise error from direct_cleanup_error
            if group_cleanup_error is not None:
                raise error from group_cleanup_error
            raise error
        if direct_cleanup_error is not None and group_cleanup_error is None:
            group_cleanup_error = direct_cleanup_error

    if group_cleanup_error is None:
        cleanup_deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
        while time.monotonic() < cleanup_deadline:
            try:
                if not _process_group_exists(pid):
                    break
                _signal_process_group(pid, signal.SIGKILL)
            except ReleaseAcceptanceError as exc:
                group_cleanup_error = exc
                break
            time.sleep(_PROCESS_POLL_SECONDS)
        if group_cleanup_error is None:
            try:
                if _process_group_exists(pid):
                    raise ReleaseAcceptanceError(
                        "verifier process group survived forced cleanup"
                    )
            except ReleaseAcceptanceError as exc:
                group_cleanup_error = exc

    if group_cleanup_error is not None:
        raise group_cleanup_error
    return status_usage


def _capture_process(
    *,
    command: list[str],
    executable: Path,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[int, int, Any, bytes, bytes, int]:
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        executable=str(executable),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        _terminate_and_reap(process.pid, already_reaped=None)
        raise ReleaseAcceptanceError("cannot capture fixed verifier streams")
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    for label, stream in streams.items():
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)

    deadline = time.monotonic() + _VERIFIER_DEADLINE_SECONDS
    reaped: tuple[int, Any] | None = None
    failure: str | None = None
    try:
        while reaped is None or selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = "fixed verifier exceeded its execution deadline"
                break
            events = selector.select(timeout=min(_PROCESS_POLL_SECONDS, remaining))
            for key, _mask in events:
                label = str(key.data)
                try:
                    chunk = os.read(key.fd, 1 << 16)
                except BlockingIOError:
                    continue
                if not chunk:
                    stream = streams[label]
                    selector.unregister(stream)
                    stream.close()
                    continue
                buffers[label].extend(chunk)
                if len(buffers[label]) > _MAX_CAPTURE_BYTES:
                    failure = f"fixed verifier {label} exceeds capture limit"
                    break
            if failure is not None:
                break
            if reaped is None:
                waited, status, usage = _wait4(process.pid, os.WNOHANG)
                if waited == process.pid:
                    reaped = (status, usage)
            elif selector.get_map() and _process_group_exists(process.pid):
                failure = "fixed verifier left a descendant process holding its streams"
                break
    except BaseException:
        _terminate_and_reap(process.pid, already_reaped=reaped)
        raise
    finally:
        selector.close()
        for stream in streams.values():
            if not stream.closed:
                stream.close()

    if failure is not None:
        _terminate_and_reap(process.pid, already_reaped=reaped)
        raise ReleaseAcceptanceError(failure)
    if reaped is None:
        raise ReleaseAcceptanceError("fixed verifier was not reaped")
    if _process_group_exists(process.pid):
        _terminate_and_reap(process.pid, already_reaped=reaped)
        raise ReleaseAcceptanceError("fixed verifier left descendant processes")
    status, exact_usage = reaped
    process.returncode = os.waitstatus_to_exitcode(status)
    finished = time.perf_counter_ns()
    return (
        process.returncode,
        process.pid,
        exact_usage,
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
        finished - started,
    )


def _run_fixed_verifier(
    *,
    archive_name: str,
    archive_payload: bytes,
    sidecar_payload: bytes,
    verifier_payload: bytes,
    interpreter_payload: bytes,
) -> dict[str, Any]:
    if not hasattr(os, "wait4"):
        raise ReleaseAcceptanceError("formal acceptance requires POSIX wait4")
    if PurePosixPath(archive_name).name != archive_name or not archive_name.endswith(".zip"):
        raise ReleaseAcceptanceError("sealed archive name is unsafe")

    temporary_name: str | None = None
    captured: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="thermoroute-same-host-release-acceptance-") as name:
        temporary_name = name
        temporary_root = Path(name).resolve()
        sealed_inputs = temporary_root / "sealed-inputs"
        sealed_inputs.mkdir(mode=0o700)
        sealed_inputs.chmod(0o700)
        sealed_python = _write_sealed_payload(
            sealed_inputs,
            name="python-bound",
            payload=interpreter_payload,
            mode=0o500,
            label="Python executable",
        )
        sealed_verifier = _write_sealed_payload(
            sealed_inputs,
            name="verify_release.py",
            payload=verifier_payload,
            mode=0o400,
            label="release verifier",
        )
        _write_sealed_payload(
            sealed_inputs,
            name=archive_name,
            payload=archive_payload,
            mode=0o400,
            label="release archive",
        )
        _write_sealed_payload(
            sealed_inputs,
            name=archive_name + ".sha256",
            payload=sidecar_payload,
            mode=0o400,
            label="archive checksum sidecar",
        )
        cwd = temporary_root / "fresh-cwd"
        cwd.mkdir(mode=0o700)
        if any(cwd.iterdir()):
            raise ReleaseAcceptanceError("fresh verifier cwd was not initially empty")
        environment = _child_environment(temporary_root)
        sealed_archive_argument = f"../sealed-inputs/{archive_name}"
        command = [
            str(sealed_python),
            "-I",
            "-B",
            str(sealed_verifier),
            sealed_archive_argument,
            "--distribution",
            LOCAL_DISTRIBUTION,
        ]
        returncode, pid, exact_usage, stdout, stderr, wall_time_ns = _capture_process(
            command=command,
            executable=sealed_python,
            cwd=cwd,
            environment=environment,
        )
        if any(cwd.iterdir()):
            raise ReleaseAcceptanceError("fixed verifier wrote into its fresh cwd")
        try:
            stdout_text = stdout.decode("utf-8")
            stderr.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseAcceptanceError("fixed verifier emitted non-UTF-8 output") from exc
        expected_line = (
            "LOCAL EVIDENCE OK [DO NOT DISTRIBUTE; PREOPEN_NOT_COMPLETE]: "
            f"{sealed_archive_argument}"
        )
        stdout_lines = stdout_text.splitlines()
        if (
            returncode != 0
            or stderr != b""
            or len(stdout_lines) != 2
            or _MANIFEST_TRANSCRIPT_LINE.fullmatch(stdout_lines[0]) is None
            or stdout_lines[1] != expected_line
            or stdout != ("\n".join(stdout_lines) + "\n").encode("utf-8")
        ):
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            raise ReleaseAcceptanceError(
                "fixed PREOPEN release verifier did not produce its exact PASS: "
                f"returncode={returncode}, detail={detail[-4000:]}"
            )
        user_ns = _seconds_to_ns(
            exact_usage.ru_utime,
            label="wait4 process-tree user CPU",
        )
        system_ns = _seconds_to_ns(
            exact_usage.ru_stime,
            label="wait4 process-tree system CPU",
        )
        raw_peak = int(exact_usage.ru_maxrss)
        raw_unit, peak_bytes = _rss_contract(raw_peak)
        captured = {
            "execution": {
                "command": list(NORMALISED_COMMAND),
                "environment": dict(CHILD_ENVIRONMENT_CONTRACT),
                "controller_pid": os.getpid(),
                "verifier_pid": pid,
                "python_isolated": True,
                "bytecode_disabled": True,
                "fresh_process": pid != os.getpid(),
                "fresh_working_directory": True,
                "fresh_temporary_extraction_root": True,
                "fresh_cwd_empty_before": True,
                "fresh_cwd_empty_after": True,
                # Set to true only after leaving TemporaryDirectory below.
                "temporary_root_removed_after_exit": False,
                "sealed_archive_argument": sealed_archive_argument,
                "process_group_cleanup_enforced": True,
                "returncode": returncode,
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stdout_bytes": len(stdout),
                "stdout_transcript": [stdout_lines[0], PROFILE_EVIDENCE_LINE],
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "stderr_bytes": len(stderr),
                "stderr_transcript": [],
                "profile_evidence_line": PROFILE_EVIDENCE_LINE,
            },
            "resource_observation": {
                "wall_time_ns": wall_time_ns,
                "user_cpu_time_ns": user_ns,
                "system_cpu_time_ns": system_ns,
                "total_cpu_time_ns": user_ns + system_ns,
                "peak_rss_raw": raw_peak,
                "peak_rss_raw_unit": raw_unit,
                "peak_rss_bytes": peak_bytes,
                "measurement_backend": MEASUREMENT_BACKEND,
                "measurement_scope": MEASUREMENT_SCOPE,
                "repetitions": 1,
            },
        }
    if temporary_name is None or os.path.lexists(temporary_name):
        raise ReleaseAcceptanceError("fresh verifier temporary root was not removed")
    captured["execution"]["temporary_root_removed_after_exit"] = True
    return captured


def _archive_material(
    root: Path,
    archive_path: str | Path,
) -> tuple[dict[str, object], bytes, bytes]:
    relative, archive = _relative_existing(root, archive_path, label="release archive")
    if PurePosixPath(relative).parent != PurePosixPath("dist") or archive.suffix != ".zip":
        raise ReleaseAcceptanceError("release archive must be one canonical dist/*.zip")
    payload = _read_regular_bytes(archive, label="release archive")
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(archive) + ".sha256")
    sidecar_relative, sidecar = _relative_existing(
        root,
        sidecar,
        label="archive checksum sidecar",
    )
    sidecar_payload = _read_regular_bytes(sidecar, label="archive checksum sidecar")
    sidecar_binding = _payload_binding(sidecar_relative, sidecar_payload)
    expected_sidecar = f"{digest}  {archive.name}\n".encode("utf-8")
    if sidecar_payload != expected_sidecar:
        raise ReleaseAcceptanceError("archive checksum sidecar is noncanonical or stale")
    return (
        {
            "path": relative,
            "sha256": digest,
            "bytes": len(payload),
            "sidecar": sidecar_binding,
        },
        payload,
        sidecar_payload,
    )


def _archive_binding(root: Path, archive_path: str | Path) -> dict[str, object]:
    return _archive_material(root, archive_path)[0]


def _runtime_material() -> tuple[dict[str, object], bytes]:
    interpreter, payload = _interpreter_material()
    return (
        {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_executable": interpreter,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
        },
        payload,
    )


def _runtime_document() -> dict[str, object]:
    return _runtime_material()[0]


def _require_keys(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ReleaseAcceptanceError(
            f"{label} exact schema changed: expected={sorted(expected)}, actual={actual}"
        )
    return value


def _valid_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if not _valid_nonnegative_int(value):
        raise ReleaseAcceptanceError(f"{label} must be a nonnegative integer")
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _validate_repository_binding(
    root: Path,
    value: object,
    *,
    expected_path: str,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    binding = _require_keys(value, _BINDING_FIELDS, label=f"{label} binding")
    if binding.get("path") != expected_path:
        raise ReleaseAcceptanceError(f"{label} binding names another path")
    path = _inside(root, expected_path, label=label)
    payload = _read_regular_bytes(path, label=label)
    if binding.get("sha256") != hashlib.sha256(payload).hexdigest() or binding.get("bytes") != len(
        payload
    ):
        raise ReleaseAcceptanceError(f"{label} byte binding changed")
    return binding, payload


def _validate_archive_document(
    root: Path,
    value: object,
) -> tuple[bytes, bytes]:
    archive = _require_keys(value, _ARCHIVE_FIELDS, label="archive")
    relative = archive.get("path")
    if not isinstance(relative, str):
        raise ReleaseAcceptanceError("archive path is malformed")
    expected, archive_payload, sidecar_payload = _archive_material(
        root,
        _inside(root, relative, label="archive"),
    )
    if dict(archive) != expected:
        raise ReleaseAcceptanceError("archive/sidecar binding changed")
    return archive_payload, sidecar_payload


def _validate_runtime(value: object) -> bytes:
    runtime = _require_keys(value, _RUNTIME_FIELDS, label="runtime")
    if any(
        runtime.get(key) != expected
        for key, expected in (
            ("python_implementation", platform.python_implementation()),
            ("python_version", platform.python_version()),
            ("platform_system", platform.system()),
            ("platform_release", platform.release()),
            ("machine", platform.machine()),
        )
    ):
        raise ReleaseAcceptanceError("receipt runtime differs from this same host")
    executable = _require_keys(
        runtime.get("python_executable"),
        _PYTHON_FIELDS,
        label="Python executable",
    )
    invoked, realpath = executable.get("invoked_path"), executable.get("realpath")
    if not isinstance(invoked, str) or not isinstance(realpath, str):
        raise ReleaseAcceptanceError("Python executable paths are malformed")
    invoked_path, real_path = Path(invoked), Path(realpath)
    if not invoked_path.is_absolute() or invoked_path.resolve() != real_path:
        raise ReleaseAcceptanceError("Python executable path binding changed")
    payload = _read_regular_bytes(real_path, label="bound Python executable")
    if executable.get("sha256") != hashlib.sha256(payload).hexdigest() or executable.get(
        "bytes"
    ) != len(payload):
        raise ReleaseAcceptanceError("Python executable byte binding changed")
    current, current_payload = _interpreter_material()
    if dict(executable) != current or payload != current_payload:
        raise ReleaseAcceptanceError("receipt Python executable differs from this validator")
    return payload


def _validate_execution(
    root: Path,
    archive_document: Mapping[str, Any],
    value: object,
) -> None:
    execution = _require_keys(value, _EXECUTION_FIELDS, label="execution")
    if (
        execution.get("command") != NORMALISED_COMMAND
        or execution.get("environment") != CHILD_ENVIRONMENT_CONTRACT
        or execution.get("python_isolated") is not True
        or execution.get("bytecode_disabled") is not True
        or execution.get("fresh_process") is not True
        or execution.get("fresh_working_directory") is not True
        or execution.get("fresh_temporary_extraction_root") is not True
        or execution.get("fresh_cwd_empty_before") is not True
        or execution.get("fresh_cwd_empty_after") is not True
        or execution.get("temporary_root_removed_after_exit") is not True
        or execution.get("process_group_cleanup_enforced") is not True
        or execution.get("returncode") != 0
        or execution.get("profile_evidence_line") != PROFILE_EVIDENCE_LINE
    ):
        raise ReleaseAcceptanceError("fixed fresh-process execution contract changed")
    controller, verifier = execution.get("controller_pid"), execution.get("verifier_pid")
    if (
        type(controller) is not int
        or controller <= 0
        or type(verifier) is not int
        or verifier <= 0
        or controller == verifier
    ):
        raise ReleaseAcceptanceError("fresh-process PID evidence is malformed")
    archive_relative = archive_document.get("path")
    if not isinstance(archive_relative, str):
        raise ReleaseAcceptanceError("execution archive path is malformed")
    _inside(root, archive_relative, label="execution archive")
    sealed_archive_argument = (
        "../sealed-inputs/" + PurePosixPath(archive_relative).name
    )
    if execution.get("sealed_archive_argument") != sealed_archive_argument:
        raise ReleaseAcceptanceError("sealed archive execution argument changed")
    stdout_transcript = execution.get("stdout_transcript")
    stderr_transcript = execution.get("stderr_transcript")
    if (
        not isinstance(stdout_transcript, list)
        or len(stdout_transcript) != 2
        or not all(isinstance(line, str) for line in stdout_transcript)
        or _MANIFEST_TRANSCRIPT_LINE.fullmatch(stdout_transcript[0]) is None
        or stdout_transcript[1] != PROFILE_EVIDENCE_LINE
        or stderr_transcript != []
    ):
        raise ReleaseAcceptanceError("verifier transcript schema changed")
    expected_stdout = (
        stdout_transcript[0]
        + "\n"
        + PROFILE_EVIDENCE_LINE.replace("<archive>", sealed_archive_argument)
        + "\n"
    ).encode("utf-8")
    expected_streams = {"stdout": expected_stdout, "stderr": b""}
    for stream, expected_payload in expected_streams.items():
        digest = execution.get(f"{stream}_sha256")
        size = execution.get(f"{stream}_bytes")
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or not _valid_nonnegative_int(size)
        ):
            raise ReleaseAcceptanceError(f"captured {stream} evidence is malformed")
        assert isinstance(size, int) and not isinstance(size, bool)
        if size > _MAX_CAPTURE_BYTES:
            raise ReleaseAcceptanceError(f"captured {stream} evidence is too large")
        if digest != hashlib.sha256(expected_payload).hexdigest() or size != len(expected_payload):
            raise ReleaseAcceptanceError(
                f"captured {stream} digest differs from its canonical transcript"
            )


def _validate_resource_observation(value: object) -> None:
    observation = _require_keys(value, _RESOURCE_FIELDS, label="resource observation")
    integer_fields = (
        "wall_time_ns",
        "user_cpu_time_ns",
        "system_cpu_time_ns",
        "total_cpu_time_ns",
        "peak_rss_raw",
        "peak_rss_bytes",
    )
    values = {
        field: _require_nonnegative_int(
            observation.get(field), label=f"resource observation {field}"
        )
        for field in integer_fields
    }
    wall = values["wall_time_ns"]
    user = values["user_cpu_time_ns"]
    system = values["system_cpu_time_ns"]
    total = values["total_cpu_time_ns"]
    raw = values["peak_rss_raw"]
    if wall == 0:
        raise ReleaseAcceptanceError("resource observation has zero wall time")
    if total != user + system:
        raise ReleaseAcceptanceError("resource CPU total is inconsistent")
    expected_unit, expected_bytes = _rss_contract(raw)
    if (
        observation.get("peak_rss_raw_unit") != expected_unit
        or observation.get("peak_rss_bytes") != expected_bytes
        or observation.get("measurement_backend") != MEASUREMENT_BACKEND
        or observation.get("measurement_scope") != MEASUREMENT_SCOPE
        or observation.get("repetitions") != 1
    ):
        raise ReleaseAcceptanceError("resource measurement contract changed")


def _write_create_only(root: Path, relative: str, payload: bytes) -> None:
    relative = _normalise_relative(relative, label="receipt")
    if relative != RECEIPT_RELATIVE:
        raise ReleaseAcceptanceError(f"receipt must use its canonical path: {RECEIPT_RELATIVE}")
    parts = PurePosixPath(relative).parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = [os.open(root, directory_flags)]
    try:
        for part in parts[:-1]:
            parent_descriptor = descriptors[-1]
            try:
                os.mkdir(part, mode=0o700, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except FileExistsError:
                pass
            try:
                child_descriptor = os.open(part, directory_flags, dir_fd=parent_descriptor)
            except OSError as exc:
                raise ReleaseAcceptanceError(
                    f"receipt parent is not one real directory: {part}"
                ) from exc
            metadata = os.fstat(child_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child_descriptor)
                raise ReleaseAcceptanceError(f"receipt parent is not a directory: {part}")
            descriptors.append(child_descriptor)

        path = root.joinpath(*parts)
        parent_descriptor = descriptors[-1]
        name = parts[-1]
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, 0o444, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise ReleaseAcceptanceError(f"refusing to replace acceptance receipt: {path}") from exc
        except OSError as exc:
            raise ReleaseAcceptanceError(f"cannot create acceptance receipt: {path}") from exc
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def run_release_mechanics_acceptance(
    archive_path: str | Path,
    *,
    root: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one fixed PREOPEN verifier child and create its canonical receipt."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ReleaseAcceptanceError("repository root is absent")
    receipt = (
        _inside(root, RECEIPT_RELATIVE, label="receipt")
        if receipt_path is None
        else _relative_existing(root, receipt_path, label="receipt")[1]
    )
    if receipt.relative_to(root).as_posix() != RECEIPT_RELATIVE:
        raise ReleaseAcceptanceError(f"receipt must use its canonical path: {RECEIPT_RELATIVE}")
    if os.path.lexists(receipt):
        raise ReleaseAcceptanceError(f"refusing to replace acceptance receipt: {receipt}")

    archive_before, archive_payload, sidecar_payload = _archive_material(root, archive_path)
    archive = _inside(root, str(archive_before["path"]), label="archive")
    source_before, source_payloads = _source_material(root, require_clean=True)
    runtime_before, interpreter_payload = _runtime_material()
    measured = _run_fixed_verifier(
        archive_name=archive.name,
        archive_payload=archive_payload,
        sidecar_payload=sidecar_payload,
        verifier_payload=source_payloads["verifier"],
        interpreter_payload=interpreter_payload,
    )

    # Close every mutable input race before publishing the receipt.  The fixed
    # verifier must also leave the repository clean.
    if _archive_binding(root, archive) != archive_before:
        raise ReleaseAcceptanceError("archive changed during fresh-process verification")
    if _source_identity(root, require_clean=True) != source_before:
        raise ReleaseAcceptanceError("source/Git identity changed during verification")
    if _runtime_document() != runtime_before:
        raise ReleaseAcceptanceError("Python/runtime identity changed during verification")

    document: dict[str, Any] = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "profile": RELEASE_PROFILE,
        "distribution": LOCAL_DISTRIBUTION,
        "evidence_scope": EVIDENCE_SCOPE,
        "archive": archive_before,
        "source": source_before,
        "runtime": runtime_before,
        "execution": measured["execution"],
        "resource_observation": measured["resource_observation"],
        "isolation_limitations": dict(ISOLATION_LIMITATIONS),
        "authentication_limitations": dict(AUTHENTICATION_LIMITATIONS),
        "label_safety": dict(LABEL_SAFETY),
    }
    document["receipt_self_sha256"] = _sha256_json(document)
    _write_create_only(root, RECEIPT_RELATIVE, _canonical_json_bytes(document))
    validated = validate_release_mechanics_receipt(receipt, root=root)
    if validated != document:
        raise ReleaseAcceptanceError("published receipt differs from exact validation")
    return document


def validate_release_mechanics_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Validate canonical bytes, exact schema, bindings, and honest claim scope."""
    root = Path(root).resolve()
    relative, receipt = _relative_existing(root, receipt_path, label="receipt")
    if relative != RECEIPT_RELATIVE:
        raise ReleaseAcceptanceError(f"receipt must use its canonical path: {RECEIPT_RELATIVE}")
    payload = _read_regular_bytes(receipt, label="acceptance receipt")
    document = _load_json_bytes(payload, label="acceptance receipt")
    if payload != _canonical_json_bytes(document):
        raise ReleaseAcceptanceError("acceptance receipt is not canonical JSON")
    _require_keys(document, _TOP_LEVEL_FIELDS, label="receipt")
    stable = dict(document)
    claimed = stable.pop("receipt_self_sha256", None)
    if not isinstance(claimed, str) or claimed != _sha256_json(stable):
        raise ReleaseAcceptanceError("acceptance receipt self-hash changed")
    if (
        document.get("format") != RECEIPT_FORMAT
        or document.get("status") != RECEIPT_STATUS
        or document.get("profile") != RELEASE_PROFILE
        or document.get("distribution") != LOCAL_DISTRIBUTION
        or document.get("evidence_scope") != EVIDENCE_SCOPE
    ):
        raise ReleaseAcceptanceError("receipt status/profile/scope changed")
    archive_document = _require_keys(document.get("archive"), _ARCHIVE_FIELDS, label="archive")
    archive_payload, sidecar_payload = _validate_archive_document(root, archive_document)
    source = _require_keys(document.get("source"), _SOURCE_FIELDS, label="source")
    source_payloads = _validate_source_identity(root, source)
    interpreter_payload = _validate_runtime(document.get("runtime"))
    _validate_execution(root, archive_document, document.get("execution"))
    _validate_resource_observation(document.get("resource_observation"))
    if document.get("isolation_limitations") != ISOLATION_LIMITATIONS:
        raise ReleaseAcceptanceError("receipt overstates process/machine isolation")
    if document.get("authentication_limitations") != AUTHENTICATION_LIMITATIONS:
        raise ReleaseAcceptanceError("receipt overstates historical/external authentication")
    if document.get("label_safety") != LABEL_SAFETY:
        raise ReleaseAcceptanceError("receipt overstates outcome-read safety evidence")
    archive_relative = archive_document.get("path")
    if not isinstance(archive_relative, str):
        raise ReleaseAcceptanceError("archive path is malformed")
    replay = _run_fixed_verifier(
        archive_name=PurePosixPath(archive_relative).name,
        archive_payload=archive_payload,
        sidecar_payload=sidecar_payload,
        verifier_payload=source_payloads["verifier"],
        interpreter_payload=interpreter_payload,
    )
    recorded_execution = _require_keys(
        document.get("execution"),
        _EXECUTION_FIELDS,
        label="execution",
    )
    replay_execution = replay["execution"]
    replay_fields = _EXECUTION_FIELDS - {"controller_pid", "verifier_pid"}
    if any(recorded_execution.get(field) != replay_execution.get(field) for field in replay_fields):
        raise ReleaseAcceptanceError(
            "receipt transcript differs from a current fixed replay of its bound archive"
        )
    return document


__all__ = [
    "AUTHENTICATION_LIMITATIONS",
    "CHILD_ENVIRONMENT_CONTRACT",
    "CORE_RELATIVE",
    "EVIDENCE_SCOPE",
    "ISOLATION_LIMITATIONS",
    "LABEL_SAFETY",
    "LOCAL_DISTRIBUTION",
    "MEASUREMENT_BACKEND",
    "MEASUREMENT_SCOPE",
    "PROFILE_EVIDENCE_LINE",
    "RECEIPT_FORMAT",
    "RECEIPT_RELATIVE",
    "RECEIPT_STATUS",
    "RELEASE_PROFILE",
    "RUNNER_RELATIVE",
    "ReleaseAcceptanceError",
    "VERIFIER_RELATIVE",
    "run_release_mechanics_acceptance",
    "validate_release_mechanics_receipt",
]
