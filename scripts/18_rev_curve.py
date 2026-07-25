#!/usr/bin/env python3
"""Publish Route-A's fail-closed development REV status.

No cost/loss ratio was preregistered, and management costs, losses, actions, and
utilities were not observed.  Moreover, Stage 16 closes raw event heads and
bundle-frozen calibrators but does not publish a receipt-bound row-level table of
the calibrated development probabilities.  Stage 18 therefore must not rebuild
Platt calibration or climatology from prediction/panel rows.

This stage reads only the already-validated Stage-16 completion receipt.  It
replaces any legacy numeric REV CSV/report/figure with explicit NOT EVALUATED
markers and writes a self-hashed status receipt last.  None of these artifacts
is eligible for the formal model suite or the opening decision.
"""
# ruff: noqa: E402
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermoroute import config as C
from thermoroute.decision import (
    REV_NOT_EVALUATED_STATUS,
    REV_STAGE16_ARTIFACT_PATHS,
    REV_STATUS_ARTIFACT_PATHS,
    REV_STATUS_FORMAT,
    REV_STATUS_PATH,
    build_rev_not_evaluated_receipt,
    rev_status_csv_bytes,
    validate_rev_not_evaluated_receipt,
)
from thermoroute.model_suite import (
    STAGE16_COMPLETION_RECEIPT_PATH,
    validate_stage16_completion_receipt,
)
from thermoroute.repro import (
    advisory_file_lock,
    atomic_write_bytes,
    atomic_write_json,
    sha256_json,
)


STAGE16_RECEIPT = ROOT / STAGE16_COMPLETION_RECEIPT_PATH
STATUS_RECEIPT = ROOT / REV_STATUS_PATH
STATUS_CSV = ROOT / REV_STATUS_ARTIFACT_PATHS["status_csv"]
STATUS_REPORT = ROOT / REV_STATUS_ARTIFACT_PATHS["status_report"]
STATUS_FIGURE = ROOT / REV_STATUS_ARTIFACT_PATHS["status_figure"]


class Stage18BoundaryError(RuntimeError):
    """Stage 18 cannot publish a trustworthy NOT EVALUATED status."""


def _binding(path: Path) -> dict[str, str]:
    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents or not resolved.is_file():
        raise Stage18BoundaryError(f"Stage-18 artifact is missing or outside root: {path}")
    return {
        "path": resolved.relative_to(resolved_root).as_posix(),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _stage16_parent_bindings() -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Read only the completion receipt; never open prediction or panel rows."""
    try:
        receipt_bytes = STAGE16_RECEIPT.read_bytes()
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage18BoundaryError("cannot read the validated Stage-16 receipt") from exc
    if not isinstance(receipt, dict):
        raise Stage18BoundaryError("Stage-16 receipt is not a JSON object")
    stable = {
        key: value for key, value in receipt.items()
        if key != "receipt_self_sha256"
    }
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("format") != "thermoroute.stage16-completion-receipt.v1"
        or receipt.get("status") != "PASS_FORMAL_STAGE16_COMPLETE"
        or receipt.get("confirmation_outcomes_requested_or_read") is not False
        or receipt.get("receipt_self_sha256") != sha256_json(stable)
        or not isinstance(artifacts, Mapping)
        or receipt.get("artifact_closure_sha256") != sha256_json(artifacts)
    ):
        raise Stage18BoundaryError("Stage-16 receipt changed after entrypoint validation")
    parent_artifacts: dict[str, dict[str, str]] = {}
    for label, expected_path in REV_STAGE16_ARTIFACT_PATHS.items():
        value = artifacts.get(label)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"path", "sha256"}
            or value.get("path") != expected_path
            or not isinstance(value.get("sha256"), str)
        ):
            raise Stage18BoundaryError(
                f"Stage-16 receipt has a noncanonical {label} binding"
            )
        parent_artifacts[label] = {
            "path": str(value["path"]),
            "sha256": str(value["sha256"]),
        }
    parent_receipt = {
        "path": STAGE16_RECEIPT.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    return parent_receipt, parent_artifacts


def _status_report() -> bytes:
    lines = [
        "# REV not evaluated",
        "",
        f"Status: `{REV_NOT_EVALUATED_STATUS}`.",
        "",
        "No REV values were computed. No cost--loss ratios were preregistered, "
        "and this study did not observe management costs, losses, actions, or a "
        "stakeholder utility model.",
        "",
        "Stage 18 did not read panel or prediction rows and did not fit or refit "
        "an event threshold, climatological reference, Platt calibrator, or any "
        "other probability transformation. Stage 16 binds raw event heads and "
        "bundle calibration metadata, but it does not publish a receipt-validated "
        "row-level table of frozen calibrated probabilities for Stage 18.",
        "",
        "This is a development-only hypothetical-sensitivity boundary. It is "
        "ineligible for the formal model suite and has no effect on the Route-A "
        "opening decision.",
        "",
        "The machine-verifiable status is `../tables/rev_curve_status.json`.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _status_figure() -> bytes:
    figure, axis = plt.subplots(figsize=(9.0, 4.2))
    axis.axis("off")
    axis.text(
        0.5,
        0.62,
        "REV NOT EVALUATED",
        ha="center",
        va="center",
        color="#A40000",
        fontsize=25,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.38,
        "No predeclared cost-loss ratios; no observed costs, losses, or actions",
        ha="center",
        va="center",
        color="#222222",
        fontsize=11,
    )
    axis.text(
        0.5,
        0.22,
        "Development-only status — not eligible for formal/opening decisions",
        ha="center",
        va="center",
        color="#444444",
        fontsize=10,
    )
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
        metadata={
            "Title": "REV NOT EVALUATED",
            "Description": REV_NOT_EVALUATED_STATUS,
            "Software": "ThermoRoute Route-A Stage18",
        },
    )
    plt.close(figure)
    payload = buffer.getvalue()
    if b"REV NOT EVALUATED" not in payload:
        raise Stage18BoundaryError("status figure lacks its machine-readable marker")
    return payload


def _incomplete_status() -> dict[str, Any]:
    return {
        "format": REV_STATUS_FORMAT,
        "status": "INCOMPLETE",
        "rev_computed": False,
        "inference_computed": False,
        "formal_suite_eligible": False,
        "formal_opening_decision_effect": False,
    }


def _run() -> None:
    parent_receipt, parent_artifacts = _stage16_parent_bindings()
    csv_payload = rev_status_csv_bytes()
    report_payload = _status_report()
    figure_payload = _status_figure()

    # Revoke any earlier final marker before replacing a possibly numeric legacy
    # artifact.  A crash can leave only INCOMPLETE, never an apparently valid old
    # curve paired with a current PASS receipt.
    atomic_write_json(STATUS_RECEIPT, _incomplete_status())
    atomic_write_bytes(STATUS_CSV, csv_payload)
    atomic_write_bytes(STATUS_REPORT, report_payload)
    atomic_write_bytes(STATUS_FIGURE, figure_payload)
    output_bindings = {
        label: _binding(ROOT / path)
        for label, path in REV_STATUS_ARTIFACT_PATHS.items()
    }
    status = build_rev_not_evaluated_receipt(
        stage16_completion_receipt=parent_receipt,
        stage16_completion_artifacts=parent_artifacts,
        artifacts=output_bindings,
    )
    atomic_write_json(STATUS_RECEIPT, status)
    validate_rev_not_evaluated_receipt(STATUS_RECEIPT, root=ROOT)
    if (
        hashlib.sha256(STAGE16_RECEIPT.read_bytes()).hexdigest()
        != parent_receipt["sha256"]
    ):
        raise Stage18BoundaryError("Stage-16 receipt changed during Stage 18")
    print(report_payload.decode("utf-8"))
    print(f"validated status receipt: {STATUS_RECEIPT.relative_to(ROOT)}")


def main() -> None:
    with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
        validate_stage16_completion_receipt(
            STAGE16_RECEIPT, root=ROOT, replay_bundle=False,
        )
        _run()


if __name__ == "__main__":
    main()
