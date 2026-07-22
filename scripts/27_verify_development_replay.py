#!/usr/bin/env python3
"""Run Route-A's full pre-confirmation model replay in fresh isolated children."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "data_usgs" / "confirmatory_model_suite_v1.json"
DEFAULT_RECEIPT = (
    ROOT / "outputs" / "model_replay" / "route_a_development_replay_v1.json"
)


def _child_environment(temporary_root: Path) -> dict[str, str]:
    """Build a complete allowlisted environment instead of inheriting one."""
    return {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "TMPDIR": str(temporary_root),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }


def _run_isolated_worker(
    *, suite: Path, receipt: Path, check: bool
) -> subprocess.CompletedProcess[str]:
    """Run exactly one replay with a distinct, initially empty bytecode cache."""
    with tempfile.TemporaryDirectory(
        prefix="thermoroute-development-replay-"
    ) as name:
        temporary_root = Path(name).resolve()
        pycache = temporary_root / "pycache"
        pycache.mkdir(mode=0o700)
        command = [
            sys.executable,
            "-I",
            "-X",
            f"pycache_prefix={pycache}",
            str(Path(__file__).resolve()),
            "--_isolated-worker",
            "--suite",
            str(suite.resolve()),
            "--receipt",
            str(receipt.resolve()),
        ]
        if check:
            command.append("--check")
        return subprocess.run(
            command,
            cwd=ROOT,
            env=_child_environment(temporary_root),
            text=True,
            capture_output=True,
            check=False,
        )


def _assert_isolated_worker_bootstrap() -> None:
    """Reject an internal worker not created with the fixed fresh-cache policy."""
    expected_flags = {
        "isolated": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": True,
        "dont_write_bytecode": 0,
    }
    actual_flags = {
        "isolated": int(sys.flags.isolated),
        "ignore_environment": int(sys.flags.ignore_environment),
        "no_user_site": int(sys.flags.no_user_site),
        "safe_path": bool(sys.flags.safe_path),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
    }
    temporary_value = os.environ.get("TMPDIR")
    if temporary_value is None or sys.pycache_prefix is None:
        raise RuntimeError("isolated replay worker lacks its temporary cache root")
    temporary_root = Path(temporary_value).resolve()
    pycache = Path(sys.pycache_prefix).resolve()
    if (
        actual_flags != expected_flags
        or pycache != temporary_root / "pycache"
        or not pycache.is_dir()
        or pycache == ROOT
        or ROOT in pycache.parents
        or dict(os.environ) != _child_environment(temporary_root)
        or Path(sys.argv[0]).resolve() != Path(__file__).resolve()
    ):
        raise RuntimeError("isolated replay worker bootstrap policy changed")


def _run_worker(*, suite: Path, receipt: Path, check: bool) -> int:
    """Load project code only after the controller establishes isolation."""
    try:
        _assert_isolated_worker_bootstrap()
    except RuntimeError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(ROOT / "src"))
    from thermoroute.development_replay import (
        fresh_verify_development_replay_receipt,
        run_guarded_development_replay,
        write_replay_receipt,
    )
    from thermoroute.model_suite import ModelSuiteError
    from thermoroute.train import configure_deterministic_runtime

    configure_deterministic_runtime()
    try:
        if check:
            document = fresh_verify_development_replay_receipt(
                receipt,
                root=ROOT,
                suite_path=suite,
                entrypoint_path=Path(__file__),
            )
        else:
            document = run_guarded_development_replay(
                root=ROOT,
                suite_path=suite,
                receipt_path=receipt,
                entrypoint_path=Path(__file__),
            )
            write_replay_receipt(receipt, document)
    except (ModelSuiteError, FileExistsError, ValueError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": document["status"],
        "models_replayed": len(document["models"]),
        "receipt": str(receipt),
        "confirmation_period_read": False,
    }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--_isolated-worker", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args._isolated_worker:
        return _run_worker(
            suite=args.suite, receipt=args.receipt, check=args.check
        )

    if not args.check:
        created = _run_isolated_worker(
            suite=args.suite, receipt=args.receipt, check=False
        )
        if created.returncode:
            parser.exit(created.returncode, created.stderr or created.stdout)
    checked = _run_isolated_worker(
        suite=args.suite, receipt=args.receipt, check=True
    )
    if checked.returncode:
        parser.exit(checked.returncode, checked.stderr or checked.stdout)
    print(checked.stdout, end="")
    return checked.returncode


if __name__ == "__main__":
    raise SystemExit(main())
