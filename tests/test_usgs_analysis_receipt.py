from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from thermoroute.repro import sha256_json


ROOT = Path(__file__).resolve().parents[1]


def _load_stage10():
    path = ROOT / "scripts" / "10_usgs_analysis.py"
    spec = importlib.util.spec_from_file_location("stage10_usgs_analysis_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE10 = _load_stage10()


@pytest.mark.parametrize(
    "verifier_diagnostic",
    (
        "ProbabilityContractError: cannot read probability receipt",
        "ProbabilityContractError: probability receipt is stale for current source",
    ),
    ids=("missing-receipt", "stale-receipt"),
)
def test_stage10_never_reads_tables_when_stage19_receipt_fails(
    tmp_path, monkeypatch, verifier_diagnostic
):
    tables = tmp_path / "tables"
    reports = tmp_path / "reports"
    tables.mkdir()
    # Tables deliberately exist: refusal must be caused by receipt validation,
    # not by the later file-existence guard.
    (tables / "probabilistic_scores.csv").write_text("model,horizon\nx,1\n")
    (tables / "multi_metric.csv").write_text("model,horizon\nx,1\n")
    monkeypatch.setattr(STAGE10.C, "TABLES", tables)
    monkeypatch.setattr(STAGE10.C, "REPORTS", reports)
    monkeypatch.setattr(
        STAGE10.C, "STAGE19_TRANSACTION_LOCK", tmp_path / "stage19.lock"
    )

    commands: list[list[str]] = []

    def failed_verifier(command, **kwargs):
        commands.append(command)
        assert command[:3] == [sys.executable, "-I", "-B"]
        assert command[-3:] == [
            "--check",
            "--receipt",
            str((tables / "probabilistic_evaluation_v2.json").resolve()),
        ]
        assert kwargs["cwd"] == ROOT
        assert kwargs["capture_output"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=verifier_diagnostic,
        )

    reads: list[object] = []

    def forbidden_read(*args, **kwargs):
        reads.append((args, kwargs))
        raise AssertionError("Stage 10 read a table before receipt validation")

    monkeypatch.setattr(STAGE10.subprocess, "run", failed_verifier)
    monkeypatch.setattr(STAGE10.pd, "read_csv", forbidden_read)
    with pytest.raises(STAGE10.Stage19ReceiptError, match="refusing to read"):
        STAGE10.main()
    assert len(commands) == 1
    assert reads == []
    assert not (reports / "usgs_analysis.md").exists()


def test_stage10_rejects_table_replaced_after_isolated_validation(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    tables = root / "outputs" / "tables"
    reports = root / "outputs" / "reports"
    tables.mkdir(parents=True)
    probability_path = tables / "probabilistic_scores.csv"
    point_path = tables / "multi_metric.csv"
    probability_path.write_bytes(b"model,horizon\nThermoRoute,1\n")
    point_path.write_bytes(b"model,horizon\nThermoRoute,1\n")

    def binding(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    document: dict[str, object] = {
        "artifacts": {
            "probability_scores": {"artifact": binding(probability_path)},
            "point_scores": {"artifact": binding(point_path)},
        }
    }
    document["receipt_self_sha256"] = sha256_json(document)
    receipt = tables / "probabilistic_evaluation_v2.json"
    receipt.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(STAGE10, "ROOT", root)
    monkeypatch.setattr(STAGE10.C, "TABLES", tables)
    monkeypatch.setattr(STAGE10.C, "REPORTS", reports)
    monkeypatch.setattr(
        STAGE10.C, "STAGE19_TRANSACTION_LOCK", root / "outputs" / ".stage19.lock"
    )

    def successful_but_racing_verifier(*_args, **_kwargs):
        probability_path.write_bytes(b"model,horizon\nOLD_OR_NEW,7\n")
        return SimpleNamespace(returncode=0, stdout="PASS", stderr="")

    monkeypatch.setattr(STAGE10.subprocess, "run", successful_but_racing_verifier)
    with pytest.raises(STAGE10.Stage19ReceiptError, match="bytes changed"):
        STAGE10.main()
    assert not (reports / "usgs_analysis.md").exists()
