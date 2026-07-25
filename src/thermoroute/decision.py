"""Hypothetical cost–loss sensitivity (Relative Economic Value).

A decision-maker can take a protective action against a high-temperature
exceedance event at cost C; an unprotected event costs L. With cost–loss ratio
α = C/L, the textbook action rule for a reliable probability p is to act iff
p > α. Relative Economic Value (Richardson 2000; Wilks) measures the fraction of
the perfect-forecast value a forecast captures, relative to a climatological
default:

    REV(α) = (E_clim − E_forecast) / (E_clim − E_perfect)

expressed per unit L: ``E_f/L = α·P(act) + P(miss)`` and
``E_perfect/L = s·α``, with base rate *s*.  Here the reference expense is
computed from an out-of-evaluation, seasonally varying probability supplied for
every row.  It is therefore not the simpler constant-base-rate expression
``min(α, s)``.

In this repository the costs, losses, actions, and ecological consequences are
not observed or stakeholder-elicited, and probability reliability is evaluated
only descriptively.  The calculation is therefore a hypothetical sensitivity
curve, not evidence of management value, operational benefit, or an optimal
decision rule.

The numerical helpers in this module are mathematical sensitivity primitives;
calling them does not make an analysis decision-eligible.  Route-A has no
predeclared cost/loss ratios and no observed costs, losses, or actions.  Stage 18
therefore publishes a machine-verifiable ``NOT_EVALUATED`` receipt instead of a
curve.  That receipt is intentionally ineligible for the formal model suite and
cannot alter the opening decision.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np

from .repro import sha256_json


REV_STATUS_FORMAT = "thermoroute.development-rev-status.v1"
REV_NOT_EVALUATED_STATUS = (
    "REV_NOT_EVALUATED_NO_PREDECLARED_COST_LOSS_RATIOS"
)
REV_STATUS_PATH = "outputs/tables/rev_curve_status.json"
REV_STATUS_ARTIFACT_PATHS: Mapping[str, str] = {
    "status_csv": "outputs/tables/rev_curve.csv",
    "status_report": "outputs/reports/rev_curve.md",
    "status_figure": "outputs/figures/fig_rev_curve.png",
}
REV_STAGE16_RECEIPT_PATH = "outputs/models/route_a_stage16_completion.json"
REV_STAGE16_ARTIFACT_PATHS: Mapping[str, str] = {
    "development_predictions": "outputs/predictions/usgs_predictions_v2.parquet",
    "development_prediction_sidecar": (
        "outputs/predictions/usgs_predictions_v2.parquet.meta.json"
    ),
    "components_pointer": "outputs/models/route_a_lstm_components.json",
}
REV_STATUS_CSV_FIELDS = (
    "format",
    "status",
    "scientific_role",
    "rev_computed",
    "inference_computed",
    "formal_suite_eligible",
    "formal_opening_decision_effect",
    "cost_loss_ratios_predeclared",
    "observed_costs_losses_actions_available",
    "receipt_validated_row_level_probability_available",
)
REV_STATUS_CSV_ROW: Mapping[str, str] = {
    "format": "thermoroute.development-rev-status-row.v1",
    "status": REV_NOT_EVALUATED_STATUS,
    "scientific_role": "development_only_hypothetical_sensitivity_boundary",
    "rev_computed": "false",
    "inference_computed": "false",
    "formal_suite_eligible": "false",
    "formal_opening_decision_effect": "false",
    "cost_loss_ratios_predeclared": "false",
    "observed_costs_losses_actions_available": "false",
    "receipt_validated_row_level_probability_available": "false",
}
REV_REASON_CODES = (
    "NO_PREDECLARED_COST_LOSS_RATIOS",
    "NO_OBSERVED_COSTS_LOSSES_OR_ACTIONS",
    "NO_RECEIPT_VALIDATED_ROW_LEVEL_FROZEN_PROBABILITY_ARTIFACT_AT_STAGE18",
)


class RevStatusError(RuntimeError):
    """A Route-A REV status receipt is stale, unsafe, or semantically invalid."""


def _is_binding(value: object, *, expected_path: str) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"path", "sha256"}
        and value.get("path") == expected_path
        and isinstance(value.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])) is not None
    )


def _require_binding(
    value: object, *, expected_path: str, label: str
) -> dict[str, str]:
    if not _is_binding(value, expected_path=expected_path):
        raise RevStatusError(f"{label} binding is malformed or noncanonical")
    assert isinstance(value, Mapping)
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def _fixed_status_payload(
    *,
    stage16_completion_receipt: Mapping[str, str],
    stage16_completion_artifacts: Mapping[str, Mapping[str, str]],
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    parent_receipt = _require_binding(
        stage16_completion_receipt,
        expected_path=REV_STAGE16_RECEIPT_PATH,
        label="Stage-16 completion receipt",
    )
    if set(stage16_completion_artifacts) != set(REV_STAGE16_ARTIFACT_PATHS):
        raise RevStatusError("Stage-16 completion artifact registry changed")
    parent_artifacts = {
        label: _require_binding(
            stage16_completion_artifacts[label],
            expected_path=path,
            label=f"Stage-16 {label}",
        )
        for label, path in REV_STAGE16_ARTIFACT_PATHS.items()
    }
    if set(artifacts) != set(REV_STATUS_ARTIFACT_PATHS):
        raise RevStatusError("REV status artifact registry changed")
    output_artifacts = {
        label: _require_binding(
            artifacts[label], expected_path=path, label=f"REV {label}"
        )
        for label, path in REV_STATUS_ARTIFACT_PATHS.items()
    }
    return {
        "format": REV_STATUS_FORMAT,
        "status": REV_NOT_EVALUATED_STATUS,
        "scientific_role": "development_only_hypothetical_sensitivity_boundary",
        "rev_computed": False,
        "inference_computed": False,
        "formal_suite_eligible": False,
        "formal_opening_decision_effect": False,
        "decision_context": {
            "cost_loss_ratios_predeclared": False,
            "cost_loss_ratios": [],
            "observed_costs_available": False,
            "observed_losses_available": False,
            "observed_actions_available": False,
            "stakeholder_utility_model_available": False,
        },
        "probability_chain": {
            "required_source": (
                "receipt_validated_row_level_bundle_frozen_development_probabilities"
            ),
            "receipt_validated_row_level_artifact_available_at_stage18": False,
            "probability_rows_consumed": 0,
            "panel_or_prediction_rows_read_by_stage18": False,
            "calibrator_refit_by_stage18": False,
            "climatology_refit_by_stage18": False,
        },
        "reason_codes": list(REV_REASON_CODES),
        "parent": {
            "stage16_completion_receipt": parent_receipt,
            "stage16_completion_artifacts": parent_artifacts,
        },
        "artifacts": output_artifacts,
    }


def build_rev_not_evaluated_receipt(
    *,
    stage16_completion_receipt: Mapping[str, str],
    stage16_completion_artifacts: Mapping[str, Mapping[str, str]],
    artifacts: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build the only admissible Route-A Stage-18 status document.

    No cost/loss ratio or probability row is accepted as an argument.  This is
    deliberate: without a predeclared decision contract and a receipt-bound
    row-level frozen-probability artifact, Stage 18 has nothing eligible to
    evaluate.
    """
    document = _fixed_status_payload(
        stage16_completion_receipt=stage16_completion_receipt,
        stage16_completion_artifacts=stage16_completion_artifacts,
        artifacts=artifacts,
    )
    return {**document, "receipt_self_sha256": sha256_json(document)}


def rev_status_csv_bytes() -> bytes:
    """Return a one-row status table that cannot be parsed as a REV curve."""
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(REV_STATUS_CSV_FIELDS))
    writer.writeheader()
    writer.writerow(dict(REV_STATUS_CSV_ROW))
    return buffer.getvalue().replace("\r\n", "\n").encode("utf-8")


def _read_bound_regular_file(
    root: Path, binding: Mapping[str, str], *, expected_path: str, label: str
) -> bytes:
    fixed = _require_binding(binding, expected_path=expected_path, label=label)
    root = root.resolve()
    path = root / expected_path
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RevStatusError(f"{label} is absent or unsafe") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or root not in resolved.parents
        or resolved.relative_to(root).as_posix() != expected_path
    ):
        raise RevStatusError(f"{label} is linked, non-regular, or outside the repository")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise RevStatusError(f"cannot read {label}") from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    refreshed = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity != refreshed:
        raise RevStatusError(f"{label} changed while being read")
    if hashlib.sha256(payload).hexdigest() != fixed["sha256"]:
        raise RevStatusError(f"{label} checksum changed")
    return payload


def _read_canonical_status_receipt(path: Path, root: Path) -> bytes:
    root = root.resolve()
    expected = root / REV_STATUS_PATH
    if path.resolve(strict=False) != expected:
        raise RevStatusError("REV status receipt has a noncanonical path")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RevStatusError("REV status receipt is absent or unsafe") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or root not in resolved.parents
        or resolved.relative_to(root).as_posix() != REV_STATUS_PATH
    ):
        raise RevStatusError("REV status receipt is linked, non-regular, or noncanonical")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise RevStatusError("cannot read REV status receipt") from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    refreshed = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity != refreshed:
        raise RevStatusError("REV status receipt changed while being read")
    return payload


def _validate_stage16_parent(
    payload: bytes, expected_artifacts: Mapping[str, Mapping[str, str]]
) -> None:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RevStatusError("Stage-16 completion receipt is malformed") from exc
    if not isinstance(document, dict):
        raise RevStatusError("Stage-16 completion receipt is not an object")
    stable = {
        key: value for key, value in document.items()
        if key != "receipt_self_sha256"
    }
    artifacts = document.get("artifacts")
    if (
        document.get("format") != "thermoroute.stage16-completion-receipt.v1"
        or document.get("status") != "PASS_FORMAL_STAGE16_COMPLETE"
        or document.get("confirmation_outcomes_requested_or_read") is not False
        or document.get("receipt_self_sha256") != sha256_json(stable)
        or not isinstance(artifacts, Mapping)
        or document.get("artifact_closure_sha256") != sha256_json(artifacts)
    ):
        raise RevStatusError("Stage-16 completion receipt is not a self-consistent PASS")
    for label, expected_path in REV_STAGE16_ARTIFACT_PATHS.items():
        actual = _require_binding(
            artifacts.get(label), expected_path=expected_path, label=f"Stage-16 {label}"
        )
        if actual != dict(expected_artifacts[label]):
            raise RevStatusError(f"REV status parent {label} drifted from Stage 16")


def _validate_status_csv(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise RevStatusError("REV status CSV is malformed") from exc
    if tuple(reader.fieldnames or ()) != REV_STATUS_CSV_FIELDS or rows != [
        dict(REV_STATUS_CSV_ROW)
    ]:
        raise RevStatusError("REV status CSV contains a value curve or changed schema")


def validate_rev_not_evaluated_receipt(
    value: Mapping[str, Any] | str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Stage 18's fail-closed status and its transitive Stage-16 parent.

    When ``root`` is supplied, this verifies the status artifacts and the exact
    Stage-16 receipt bytes.  It intentionally does not open the parent panel or
    prediction table; the Stage-16 gate is responsible for validating those
    artifacts before Stage 18 starts, while the copied bindings make any parent
    substitution visible here.
    """
    status_path: Path | None = None
    status_payload: bytes | None = None
    if isinstance(value, Mapping):
        document = dict(value)
    else:
        status_path = Path(value)
        try:
            status_payload = (
                _read_canonical_status_receipt(status_path, Path(root))
                if root is not None
                else status_path.read_bytes()
            )
            document = json.loads(status_payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RevStatusError("cannot read REV status receipt") from exc
        if not isinstance(document, dict):
            raise RevStatusError("REV status receipt is not an object")
    expected_fields = {
        "format",
        "status",
        "scientific_role",
        "rev_computed",
        "inference_computed",
        "formal_suite_eligible",
        "formal_opening_decision_effect",
        "decision_context",
        "probability_chain",
        "reason_codes",
        "parent",
        "artifacts",
        "receipt_self_sha256",
    }
    if set(document) != expected_fields:
        raise RevStatusError("REV status receipt schema changed")
    stable = {
        key: item for key, item in document.items()
        if key != "receipt_self_sha256"
    }
    if document.get("receipt_self_sha256") != sha256_json(stable):
        raise RevStatusError("REV status receipt self-hash changed")
    parent = document.get("parent")
    artifacts = document.get("artifacts")
    if (
        not isinstance(parent, Mapping)
        or set(parent) != {
            "stage16_completion_receipt", "stage16_completion_artifacts"
        }
        or not isinstance(parent.get("stage16_completion_artifacts"), Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise RevStatusError("REV status receipt parent/artifact closure changed")
    expected = _fixed_status_payload(
        stage16_completion_receipt=parent["stage16_completion_receipt"],
        stage16_completion_artifacts=parent["stage16_completion_artifacts"],
        artifacts=artifacts,
    )
    if stable != expected:
        raise RevStatusError("REV status semantics changed")

    if root is not None:
        resolved_root = Path(root).resolve()
        parent_receipt = _read_bound_regular_file(
            resolved_root,
            parent["stage16_completion_receipt"],
            expected_path=REV_STAGE16_RECEIPT_PATH,
            label="Stage-16 completion receipt",
        )
        _validate_stage16_parent(
            parent_receipt, parent["stage16_completion_artifacts"]
        )
        payloads = {
            label: _read_bound_regular_file(
                resolved_root,
                artifacts[label],
                expected_path=path,
                label=f"REV {label}",
            )
            for label, path in REV_STATUS_ARTIFACT_PATHS.items()
        }
        _validate_status_csv(payloads["status_csv"])
        try:
            report = payloads["status_report"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RevStatusError("REV status report is not UTF-8") from exc
        if (
            REV_NOT_EVALUATED_STATUS not in report
            or "No REV values were computed" not in report
        ):
            raise RevStatusError("REV report can be mistaken for a numeric result")
        figure = payloads["status_figure"]
        if not figure.startswith(b"\x89PNG\r\n\x1a\n") or b"REV NOT EVALUATED" not in figure:
            raise RevStatusError("REV figure is not an explicit NOT EVALUATED marker")
        if status_path is not None and status_payload is not None:
            if _read_canonical_status_receipt(status_path, resolved_root) != status_payload:
                raise RevStatusError("REV status receipt changed during validation")
    return document


def rev_curve(events: np.ndarray, score: np.ndarray, alphas: np.ndarray,
              probabilistic: bool = True,
              reference_probability: np.ndarray | None = None) -> np.ndarray:
    """Relative Economic Value over a grid of cost–loss ratios.

    ``events``  binary outcome (1 = exceedance occurred).
    ``score``   probability score (probabilistic=True) or a fixed binary
                warning (probabilistic=False, e.g. persistence > threshold).
    """
    events = np.asarray(events, dtype=int)
    score = np.asarray(score, dtype=float)
    if reference_probability is None:
        raise ValueError(
            "reference_probability must be fitted outside the evaluation sample"
        )
    reference_probability = np.asarray(reference_probability, dtype=float)
    if reference_probability.shape != events.shape:
        raise ValueError("reference_probability must align one-to-one with events")
    s = float(events.mean())
    rev = np.empty_like(alphas, dtype=float)
    for i, a in enumerate(alphas):
        act = (score > a) if probabilistic else (score > 0.5)
        p_act = float(act.mean())
        miss = float(((~act) & (events == 1)).mean())
        e_f = a * p_act + miss
        ref_act = reference_probability > a
        e_clim = a * float(ref_act.mean()) + float(((~ref_act) & (events == 1)).mean())
        e_perf = s * a
        denom = e_clim - e_perf
        rev[i] = (e_clim - e_f) / denom if denom > 1e-12 else np.nan
    return rev


def value_summary(events: np.ndarray, score: np.ndarray,
                  probabilistic: bool = True,
                  reference_probability: np.ndarray | None = None,
                  report_alphas=(0.05, 0.10, 0.20, 0.50)) -> dict:
    """Peak REV (and the α achieving it) plus REV at representative α values."""
    grid = np.linspace(0.01, 0.99, 99)
    rev = rev_curve(events, score, grid, probabilistic, reference_probability)
    out = {"base_rate": float(events.mean()),
           "REV_max": float(np.nanmax(rev)),
           "alpha_at_max": float(grid[int(np.nanargmax(rev))])}
    for a in report_alphas:
        r = rev_curve(events, score, np.array([a]), probabilistic,
                      reference_probability)[0]
        out[f"REV@{a:g}"] = float(r)
    return out


def cluster_bootstrap_rev(events: np.ndarray, score: np.ndarray,
                          reference_probability: np.ndarray,
                          clusters: np.ndarray, alpha: float, *,
                          probabilistic: bool = True, n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float]:
    """Percentile interval resampling whole spatial/hydrologic clusters."""
    events = np.asarray(events)
    score = np.asarray(score)
    reference_probability = np.asarray(reference_probability)
    clusters = np.asarray(clusters)
    if not (len(events) == len(score) == len(reference_probability) == len(clusters)):
        raise ValueError("events, scores, references, and clusters must align")
    unique = np.unique(clusters)
    if len(unique) < 2:
        return float("nan"), float("nan")
    by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_cluster[cluster] for cluster in sampled])
        value = rev_curve(
            events[index], score[index], np.array([alpha]), probabilistic,
            reference_probability[index],
        )[0]
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
