from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ENTRYPOINTS = (
    ("scripts/09_usgs_experiment.py", "--_thermoroute-stage09-worker"),
    ("scripts/09b_development_controls.py", "--_thermoroute-stage09b-worker"),
    ("scripts/16_lstm_baseline.py", "--_thermoroute-stage16-worker"),
    ("scripts/24_freeze_model_suite.py", "--_thermoroute-stage24-worker"),
    ("scripts/25_train_external_pooled_suite.py", "--_thermoroute-stage25-worker"),
)

RUNTIME_ENFORCED_ENTRYPOINTS = {
    "scripts/09_usgs_experiment.py": 4,
    "scripts/09b_development_controls.py": 3,
    "scripts/16_lstm_baseline.py": 3,
    "scripts/25_train_external_pooled_suite.py": 4,
    "scripts/27_verify_development_replay.py": 1,
}

FINAL_PUBLICATION_GUARDS = {
    "scripts/09_usgs_experiment.py": {"publish_stage09_completion_receipt"},
    "scripts/09b_development_controls.py": {
        "publish_stage09b_completion_receipt"
    },
    "scripts/16_lstm_baseline.py": {"write_component_pointer"},
    "scripts/25_train_external_pooled_suite.py": {
        "publish_stage25_completion_receipt"
    },
    "scripts/27_verify_development_replay.py": {"write_replay_receipt"},
}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_native_policy_assertion(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _call_name(statement.value) == "assert_formal_numerical_policy"
    )


@pytest.mark.parametrize(
    ("relative", "publishers"),
    FINAL_PUBLICATION_GUARDS.items(),
)
def test_final_receipt_publication_has_same_block_native_policy_guard(
    relative, publishers,
):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    parent: dict[ast.AST, ast.AST] = {}
    statement_block: dict[ast.stmt, list[ast.stmt]] = {}
    for node in ast.walk(tree):
        for _field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if isinstance(child, ast.AST):
                    parent[child] = node
            if isinstance(value, list) and all(
                isinstance(child, ast.stmt) for child in value
            ):
                for child in value:
                    statement_block[child] = value

    found: set[str] = set()
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        name = _call_name(call)
        if name not in publishers:
            continue
        found.add(name)
        statement: ast.AST = call
        while not isinstance(statement, ast.stmt):
            statement = parent[statement]
        block = statement_block[statement]
        position = block.index(statement)
        assert any(
            _is_native_policy_assertion(item)
            for item in block[max(0, position - 2):position]
        )
    assert found == publishers


@pytest.mark.parametrize(
    ("relative", "minimum_assertions"),
    RUNTIME_ENFORCED_ENTRYPOINTS.items(),
)
def test_formal_numeric_entrypoints_use_full_native_runtime_contract(
    relative, minimum_assertions,
):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "thermoroute.repro", "configure_deterministic_runtime"
    ) in imports
    assert (
        "thermoroute.train", "configure_deterministic_runtime"
    ) not in imports
    assertions = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_formal_numerical_policy"
        for node in ast.walk(tree)
    )
    assert assertions >= minimum_assertions


def test_stage09_formal_publication_gate_is_cpu_only():
    path = ROOT / "scripts" / "09_usgs_experiment.py"
    spec = importlib.util.spec_from_file_location("stage09_cpu_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    canonical = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
    assert module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=False
    )
    for device in ("auto", "mps", "cuda"):
        assert not module.formal_publication_candidate(
            panel_path=canonical, training_device=device, exploratory=False
        )
    assert not module.formal_publication_candidate(
        panel_path=canonical, training_device="cpu", exploratory=True
    )


def _artifact_snapshot() -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for relative in (
        "outputs/runs", "outputs/models", "outputs/predictions", "outputs/tables",
    ):
        base = ROOT / relative
        if not base.exists():
            continue
        base_stat = base.stat()
        snapshot[relative] = (-1, int(base_stat.st_mtime_ns))
        for path in base.rglob("*"):
            stat = path.stat()
            if path.is_dir():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    -1, int(stat.st_mtime_ns),
                )
            elif path.is_file():
                snapshot[path.relative_to(ROOT).as_posix()] = (
                    int(stat.st_size), int(stat.st_mtime_ns),
                )
    return snapshot


@pytest.mark.parametrize(("relative", "_worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_entrypoint_help_is_zero_training_and_zero_artifact_output(
    relative, _worker_argument,
):
    before = _artifact_snapshot()
    result = subprocess.run(
        [sys.executable, str(ROOT / relative), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert _artifact_snapshot() == before


@pytest.mark.parametrize(("relative", "worker_argument"), FORMAL_ENTRYPOINTS)
def test_formal_worker_cannot_bypass_controller_handshake(
    tmp_path, relative, worker_argument,
):
    cache = tmp_path / "untrusted-cache"
    cache.mkdir()
    result = subprocess.run(
        [
            sys.executable, "-I", "-X", f"pycache_prefix={cache}",
            str(ROOT / relative), worker_argument, "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=os.environ.copy(),
    )
    assert result.returncode != 0
    assert "worker handshake is incomplete" in result.stderr
