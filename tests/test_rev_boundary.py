from __future__ import annotations

import ast
import csv
import importlib.util
from io import StringIO
import json
import os
from pathlib import Path
import sys

import pytest

from thermoroute.decision import (
    REV_NOT_EVALUATED_STATUS,
    REV_STAGE16_ARTIFACT_PATHS,
    REV_STAGE16_RECEIPT_PATH,
    REV_STATUS_ARTIFACT_PATHS,
    REV_STATUS_CSV_FIELDS,
    REV_STATUS_CSV_ROW,
    REV_STATUS_PATH,
    RevStatusError,
    validate_rev_not_evaluated_receipt,
)
from thermoroute.repro import sha256_json


ROOT = Path(__file__).resolve().parents[1]
STAGE18_PATH = ROOT / "scripts/18_rev_curve.py"


def _load_stage18():
    name = "thermoroute_test_stage18_rev_boundary"
    specification = importlib.util.spec_from_file_location(name, STAGE18_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _binding(path: str, token: str) -> dict[str, str]:
    return {"path": path, "sha256": token * 64}


def _stage16_receipt(root: Path) -> Path:
    artifacts = {
        label: _binding(path, token)
        for (label, path), token in zip(
            REV_STAGE16_ARTIFACT_PATHS.items(), ("a", "b", "c"), strict=True
        )
    }
    document: dict[str, object] = {
        "format": "thermoroute.stage16-completion-receipt.v1",
        "status": "PASS_FORMAL_STAGE16_COMPLETE",
        "confirmation_outcomes_requested_or_read": False,
        "artifacts": artifacts,
        "artifact_closure_sha256": sha256_json(artifacts),
    }
    document["receipt_self_sha256"] = sha256_json(document)
    path = root / REV_STAGE16_RECEIPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _patch_paths(module, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "STAGE16_RECEIPT", root / REV_STAGE16_RECEIPT_PATH)
    monkeypatch.setattr(module, "STATUS_RECEIPT", root / REV_STATUS_PATH)
    monkeypatch.setattr(
        module, "STATUS_CSV", root / REV_STATUS_ARTIFACT_PATHS["status_csv"]
    )
    monkeypatch.setattr(
        module, "STATUS_REPORT", root / REV_STATUS_ARTIFACT_PATHS["status_report"]
    )
    monkeypatch.setattr(
        module, "STATUS_FIGURE", root / REV_STATUS_ARTIFACT_PATHS["status_figure"]
    )


def _old_numeric_outputs(root: Path) -> dict[str, bytes]:
    payloads = {
        "status_csv": b"horizon,model,alpha,REV,REV_ci_low,REV_ci_high\n1,x,0.2,0.7,0.5,0.9\n",
        "status_report": b"REV = 0.700 at C/L = 0.20\n",
        "status_figure": b"legacy numeric REV figure\n",
    }
    for label, relative in REV_STATUS_ARTIFACT_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[label])
    return payloads


def _run_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "repo"
    _stage16_receipt(root)
    old = _old_numeric_outputs(root)
    module = _load_stage18()
    _patch_paths(module, root, monkeypatch)
    module._run()
    receipt = validate_rev_not_evaluated_receipt(
        root / REV_STATUS_PATH, root=root
    )
    return module, root, old, receipt


def test_stage18_overwrites_legacy_numeric_outputs_with_bound_not_evaluated_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, root, old, receipt = _run_fixture(tmp_path, monkeypatch)

    assert receipt["status"] == REV_NOT_EVALUATED_STATUS
    assert receipt["rev_computed"] is False
    assert receipt["inference_computed"] is False
    assert receipt["formal_suite_eligible"] is False
    assert receipt["formal_opening_decision_effect"] is False
    assert receipt["decision_context"]["cost_loss_ratios"] == []
    assert receipt["probability_chain"] == {
        "required_source": (
            "receipt_validated_row_level_bundle_frozen_development_probabilities"
        ),
        "receipt_validated_row_level_artifact_available_at_stage18": False,
        "probability_rows_consumed": 0,
        "panel_or_prediction_rows_read_by_stage18": False,
        "calibrator_refit_by_stage18": False,
        "climatology_refit_by_stage18": False,
    }
    csv_payload = (root / REV_STATUS_ARTIFACT_PATHS["status_csv"]).read_bytes()
    report = (root / REV_STATUS_ARTIFACT_PATHS["status_report"]).read_text()
    figure = (root / REV_STATUS_ARTIFACT_PATHS["status_figure"]).read_bytes()
    assert csv_payload != old["status_csv"]
    assert report.encode() != old["status_report"]
    assert figure != old["status_figure"]
    reader = csv.DictReader(StringIO(csv_payload.decode("utf-8")))
    assert tuple(reader.fieldnames or ()) == REV_STATUS_CSV_FIELDS
    assert list(reader) == [dict(REV_STATUS_CSV_ROW)]
    assert not {"REV", "alpha", "REV_ci_low", "REV_ci_high"} & set(
        reader.fieldnames or ()
    )
    assert "No REV values were computed" in report
    assert REV_NOT_EVALUATED_STATUS in report
    assert figure.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"REV NOT EVALUATED" in figure


def test_stage18_does_not_require_or_open_parent_prediction_or_panel_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, root, _old, receipt = _run_fixture(tmp_path, monkeypatch)
    # The fixture deliberately creates no parent prediction, sidecar, component
    # pointer, or panel.  Their immutable Stage-16 bindings are copied, not read.
    assert all(
        not (root / path).exists() for path in REV_STAGE16_ARTIFACT_PATHS.values()
    )
    assert receipt["parent"]["stage16_completion_artifacts"] == {
        label: _binding(path, token)
        for (label, path), token in zip(
            REV_STAGE16_ARTIFACT_PATHS.items(), ("a", "b", "c"), strict=True
        )
    }


def test_parent_artifact_binding_drift_is_rejected_even_when_status_is_resealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, root, _old, _receipt = _run_fixture(tmp_path, monkeypatch)
    status_path = root / REV_STATUS_PATH
    changed = json.loads(status_path.read_text())
    changed["parent"]["stage16_completion_artifacts"][
        "development_predictions"
    ]["sha256"] = "f" * 64
    changed.pop("receipt_self_sha256")
    changed["receipt_self_sha256"] = sha256_json(changed)
    status_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(RevStatusError, match="drifted from Stage 16"):
        validate_rev_not_evaluated_receipt(status_path, root=root)


@pytest.mark.parametrize(
    ("attack", "match"),
    (
        ("status_self_hash", "self-hash changed"),
        ("stage16_receipt", "checksum changed"),
        ("status_csv", "checksum changed"),
    ),
)
def test_status_parent_and_output_drift_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    match: str,
) -> None:
    _module, root, _old, _receipt = _run_fixture(tmp_path, monkeypatch)
    status_path = root / REV_STATUS_PATH
    if attack == "status_self_hash":
        changed = json.loads(status_path.read_text())
        changed["formal_suite_eligible"] = True
        status_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    elif attack == "stage16_receipt":
        (root / REV_STAGE16_RECEIPT_PATH).write_bytes(b"changed parent receipt\n")
    else:
        (root / REV_STATUS_ARTIFACT_PATHS["status_csv"]).write_bytes(
            b"model,alpha,REV\nThermoRoute,0.2,0.9\n"
        )
    with pytest.raises(RevStatusError, match=match):
        validate_rev_not_evaluated_receipt(status_path, root=root)


def test_interrupted_replacement_revokes_final_status_before_legacy_curve_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _stage16_receipt(root)
    old = _old_numeric_outputs(root)
    module = _load_stage18()
    _patch_paths(module, root, monkeypatch)

    def fail_before_first_artifact(_path: Path, _payload: bytes) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr(module, "atomic_write_bytes", fail_before_first_artifact)
    with pytest.raises(OSError, match="injected replacement failure"):
        module._run()
    incomplete = json.loads((root / REV_STATUS_PATH).read_text())
    assert incomplete["status"] == "INCOMPLETE"
    assert (root / REV_STATUS_ARTIFACT_PATHS["status_csv"]).read_bytes() == old[
        "status_csv"
    ]
    with pytest.raises(RevStatusError, match="schema changed"):
        validate_rev_not_evaluated_receipt(root / REV_STATUS_PATH, root=root)


def test_stage18_source_has_no_probability_refit_or_outcome_table_reader() -> None:
    tree = ast.parse(STAGE18_PATH.read_text(encoding="utf-8"), filename=str(STAGE18_PATH))
    forbidden_modules = {
        "thermoroute.data",
        "thermoroute.evidence",
        "thermoroute.probability",
        "thermoroute.results",
        "pandas",
    }
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not forbidden_modules & imports
    forbidden_calls = {
        "calibrated_event_frame",
        "fit_horizon_calibrators",
        "fit_seasonal_climatology",
        "fit_frozen_seasonal_event_reference",
        "load_route_a_predictions",
        "read_parquet",
        "read_csv",
        "rev_curve",
        "cluster_bootstrap_rev",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not forbidden_calls & calls
    source = STAGE18_PATH.read_text(encoding="utf-8")
    assert "p_exceed_calibrated" not in source
    assert "calibration_split" not in source
    assert "outputs/predictions" not in source


def test_status_artifact_symlink_and_hardlink_aliases_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _module, root, _old, _receipt = _run_fixture(tmp_path, monkeypatch)
    status_path = root / REV_STATUS_PATH
    csv_path = root / REV_STATUS_ARTIFACT_PATHS["status_csv"]
    target = csv_path.with_name("rev-status-target.csv")
    csv_path.rename(target)
    csv_path.symlink_to(target.name)
    with pytest.raises(RevStatusError, match="linked"):
        validate_rev_not_evaluated_receipt(status_path, root=root)

    csv_path.unlink()
    target.rename(csv_path)
    alias = csv_path.with_name("rev-status-hardlink.csv")
    os.link(csv_path, alias)
    with pytest.raises(RevStatusError, match="linked"):
        validate_rev_not_evaluated_receipt(status_path, root=root)


@pytest.mark.parametrize("attack", ("symlink", "hardlink"))
def test_status_receipt_link_aliases_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _module, root, _old, _receipt = _run_fixture(tmp_path, monkeypatch)
    status_path = root / REV_STATUS_PATH
    alias = status_path.with_name(f"rev-status-{attack}.json")
    if attack == "symlink":
        status_path.rename(alias)
        status_path.symlink_to(alias.name)
    else:
        os.link(status_path, alias)
    with pytest.raises(RevStatusError, match="noncanonical|linked"):
        validate_rev_not_evaluated_receipt(status_path, root=root)
