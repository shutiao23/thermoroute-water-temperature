from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONSUMERS = (
    "15_stratified.py",
    "18_rev_curve.py",
    "20_tuurt.py",
    "22_adaptive_conformal.py",
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("script_name", CONSUMERS)
def test_v2_consumer_entrypoint_has_exact_shared_stage16_gate(script_name):
    path = ROOT / "scripts" / script_name
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = _function(tree, "main")
    assert len(main.body) == 1
    transaction = main.body[0]
    assert isinstance(transaction, ast.With)
    assert len(transaction.items) == 1
    lock = transaction.items[0].context_expr
    assert isinstance(lock, ast.Call)
    assert isinstance(lock.func, ast.Name)
    assert lock.func.id == "advisory_file_lock"
    assert len(lock.args) == 1
    assert ast.unparse(lock.args[0]) == "C.STAGE16_TRANSACTION_LOCK"
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in lock.keywords
    } == {"exclusive": False}

    assert len(transaction.body) == 2
    gate_statement, run_statement = transaction.body
    assert isinstance(gate_statement, ast.Expr)
    gate = gate_statement.value
    assert isinstance(gate, ast.Call)
    assert isinstance(gate.func, ast.Name)
    assert gate.func.id == "validate_stage16_completion_receipt"
    assert [ast.unparse(argument) for argument in gate.args] == [
        "STAGE16_RECEIPT"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in gate.keywords
    } == {"root": "ROOT", "replay_bundle": "False"}
    assert isinstance(run_statement, ast.Expr)
    run = run_statement.value
    assert isinstance(run, ast.Call)
    assert isinstance(run.func, ast.Name)
    assert run.func.id == "_run"
    assert not run.args and not run.keywords


def _load_script(script_name: str):
    module_name = f"thermoroute_test_{Path(script_name).stem}"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "scripts" / script_name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


@pytest.mark.parametrize("script_name", CONSUMERS)
def test_v2_consumer_entrypoint_fails_before_read_or_publish(
    script_name, monkeypatch, tmp_path,
):
    module = _load_script(script_name)
    events: list[object] = []
    lock_path = tmp_path / f"{Path(script_name).stem}.lock"
    monkeypatch.setattr(module.C, "STAGE16_TRANSACTION_LOCK", lock_path)

    @contextmanager
    def fake_lock(path, *, exclusive):
        events.append(("lock-enter", path, exclusive))
        try:
            yield
        finally:
            events.append(("lock-exit", path, exclusive))

    def reject_gate(path, *, root, replay_bundle):
        events.append(("gate", path, root, replay_bundle))
        raise RuntimeError("missing Stage-16 receipt")

    def forbidden_run():
        events.append("run")
        raise AssertionError("V2 consumer ran before Stage-16 validation")

    monkeypatch.setattr(module, "advisory_file_lock", fake_lock)
    monkeypatch.setattr(module, "validate_stage16_completion_receipt", reject_gate)
    monkeypatch.setattr(module, "_run", forbidden_run)
    with pytest.raises(RuntimeError, match="missing Stage-16 receipt"):
        module.main()
    assert events == [
        ("lock-enter", lock_path, False),
        ("gate", module.STAGE16_RECEIPT, module.ROOT, False),
        ("lock-exit", lock_path, False),
    ]
