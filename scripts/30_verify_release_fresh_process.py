#!/usr/bin/env python3
"""Create one same-host fresh-process PREOPEN release-mechanics receipt.

The controller first re-executes itself under a fixed ``python -I -B -S``
bootstrap with a fresh, empty bytecode-cache namespace.  It verifies the core
source against the clean Git ``HEAD`` before compiling those exact source bytes;
it never imports the eager :mod:`thermoroute` package.  The sole verifier child
is a sealed copy of the Git-bound ``scripts/verify_release.py`` and has no
command override or post-opening mode.

A PASS is a local engineering observation, not external authentication, proof
of historical execution or resource use, model readiness, a scientific result,
fresh-machine isolation, or deployment performance.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CORE_RELATIVE = "src/thermoroute/release_acceptance.py"
RUNNER_RELATIVE = "scripts/30_verify_release_fresh_process.py"
_BOOTSTRAP_ROOT_ENV = "THERMOROUTE_RELEASE_ACCEPTANCE_BOOTSTRAP_ROOT"
_BOOTSTRAP_ENVIRONMENT = {
    "PATH": os.defpath,
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
}


class BootstrapError(RuntimeError):
    """The stdlib-only controller bootstrap failed closed."""


def _bootstrap_root() -> Path:
    raw = os.environ.get(_BOOTSTRAP_ROOT_ENV)
    if raw is None:
        name = tempfile.mkdtemp(prefix="thermoroute-release-controller-bootstrap-")
        root = Path(name).resolve()
        pycache = root / "empty-pycache"
        pycache.mkdir(mode=0o700)
        python = Path(sys.executable).resolve()
        environment = {
            **_BOOTSTRAP_ENVIRONMENT,
            _BOOTSTRAP_ROOT_ENV: str(root),
        }
        arguments = [
            str(python),
            "-I",
            "-B",
            "-S",
            "-X",
            f"pycache_prefix={pycache}",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        os.execve(str(python), arguments, environment)
        raise AssertionError("os.execve unexpectedly returned")

    root = Path(raw)
    expected_cache = root / "empty-pycache"
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not expected_cache.is_dir()
        or stat.S_IMODE(root.stat().st_mode) != 0o700
        or stat.S_IMODE(expected_cache.stat().st_mode) != 0o700
        or sys.flags.isolated != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.no_site != 1
        or not sys.flags.safe_path
        or sys.pycache_prefix != str(expected_cache)
        or any(expected_cache.iterdir())
        or dict(os.environ)
        != {**_BOOTSTRAP_ENVIRONMENT, _BOOTSTRAP_ROOT_ENV: str(root)}
    ):
        raise BootstrapError("controller did not enter its exact isolated empty-pycache bootstrap")
    return root


BOOTSTRAP_ROOT = _bootstrap_root()


import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from typing import Any  # noqa: E402


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BootstrapError(f"cannot open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BootstrapError(f"{label} must be one non-linked regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _git_environment() -> dict[str, str]:
    return {
        **_BOOTSTRAP_ENVIRONMENT,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git(*arguments: str) -> bytes:
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise BootstrapError("cannot find Git in the fixed system PATH")
    result = subprocess.run(
        [
            executable,
            "--no-replace-objects",
            "-c",
            "core.useReplaceRefs=false",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(ROOT),
            *arguments,
        ],
        env=_git_environment(),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise BootstrapError(f"Git bootstrap command failed: {detail[-2000:]}")
    return result.stdout


def _verified_core_source() -> bytes:
    top = _git("rev-parse", "--show-toplevel").decode().strip()
    if Path(top).resolve() != ROOT:
        raise BootstrapError("controller requires the exact Git worktree root")
    if _git("rev-parse", "--is-shallow-repository").decode().strip() != "false":
        raise BootstrapError("shallow Git history is prohibited")
    dirt = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirt:
        lines = dirt.decode("utf-8", errors="replace").splitlines()
        raise BootstrapError(f"controller requires a clean Git worktree: {lines[:10]}")
    commit = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise BootstrapError("controller Git commit is malformed")

    payloads: dict[str, bytes] = {}
    for relative in (CORE_RELATIVE, RUNNER_RELATIVE):
        current = _read_regular_bytes(ROOT / relative, label=relative)
        committed = _git("show", f"{commit}:{relative}")
        if current != committed:
            raise BootstrapError(f"{relative} differs from clean Git HEAD")
        payloads[relative] = current
    return payloads[CORE_RELATIVE]


def _load_core() -> dict[str, Any]:
    path = ROOT / CORE_RELATIVE
    payload = _verified_core_source()
    # ``compile`` consumes the exact bytes already compared with the Git blob;
    # no package import machinery or adjacent bytecode cache participates.
    namespace: dict[str, Any] = {
        "__name__": "_thermoroute_release_acceptance_sealed_core",
        "__file__": str(path),
        "__package__": None,
    }
    code = compile(payload, str(path), "exec", dont_inherit=True)
    exec(code, namespace)
    required = {
        "RECEIPT_RELATIVE",
        "ReleaseAcceptanceError",
        "run_release_mechanics_acceptance",
    }
    if not required.issubset(namespace):
        raise BootstrapError("verified acceptance core lacks its fixed interface")
    return namespace


try:
    CORE = _load_core()
except BaseException:
    shutil.rmtree(BOOTSTRAP_ROOT, ignore_errors=True)
    raise
RECEIPT_RELATIVE = str(CORE["RECEIPT_RELATIVE"])
ReleaseAcceptanceError = CORE["ReleaseAcceptanceError"]
run_release_mechanics_acceptance = CORE["run_release_mechanics_acceptance"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help=(
            "existing local-only PREOPEN_NOT_COMPLETE dist/*.zip; the exact "
            "sibling .sha256 sidecar is mandatory"
        ),
    )
    args = parser.parse_args()
    try:
        document = run_release_mechanics_acceptance(
            args.archive,
            root=ROOT,
            receipt_path=ROOT / RECEIPT_RELATIVE,
        )
    except (OSError, ReleaseAcceptanceError, ValueError) as exc:
        print(f"release-mechanics acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": document["status"],
                "profile": document["profile"],
                "evidence_scope": document["evidence_scope"],
                "archive_sha256": document["archive"]["sha256"],
                "receipt": RECEIPT_RELATIVE,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _entrypoint() -> int:
    try:
        return main()
    finally:
        shutil.rmtree(BOOTSTRAP_ROOT)


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
