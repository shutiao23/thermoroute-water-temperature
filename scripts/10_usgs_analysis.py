#!/usr/bin/env python3
"""Route-A synthesis of already generated USGS verification artifacts.

Mechanistic claims based on latent kappa or router weights were retired: neither
quantity is identifiable or causal.  Probability calibration and hypothetical
decision sensitivity are produced by stages 19 and 18 respectively.
"""
# ruff: noqa: E402
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thermoroute import config as C
from thermoroute.repro import (
    advisory_file_lock,
    atomic_write_bytes,
    sha256_json,
)


class Stage19ReceiptError(RuntimeError):
    """Stage 10 refuses to read unverified or stale Stage-19 outputs."""


def _validate_stage19_receipt(receipt: Path) -> None:
    """Validate Stage 19 in a fresh isolated interpreter before reading tables."""
    command = [
        sys.executable,
        "-I",
        "-B",
        str(ROOT / "scripts" / "19_probabilistic.py"),
        "--check",
        "--receipt",
        str(receipt.resolve()),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise Stage19ReceiptError(
            "cannot launch the isolated Stage-19 receipt verifier"
        ) from exc
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip().splitlines()
        detail = diagnostic[-1] if diagnostic else "no verifier diagnostic"
        raise Stage19ReceiptError(
            "Stage-19 receipt validation failed; refusing to read probability "
            f"or point tables ({detail})"
        )


def _read_bound_stage19_tables(
    receipt_path: Path, *, root: Path = ROOT
) -> tuple[bytes, bytes]:
    """Read each table once and verify those exact bytes against the receipt."""
    try:
        receipt_bytes = receipt_path.read_bytes()
        document = json.loads(receipt_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage19ReceiptError("cannot read the validated Stage-19 receipt") from exc
    stable = dict(document) if isinstance(document, dict) else {}
    self_hash = stable.pop("receipt_self_sha256", None)
    if self_hash != sha256_json(stable):
        raise Stage19ReceiptError("Stage-19 receipt changed after isolated validation")
    artifacts = document.get("artifacts")
    expected = {
        "probability_scores": "outputs/tables/probabilistic_scores.csv",
        "point_scores": "outputs/tables/multi_metric.csv",
    }
    payloads: dict[str, bytes] = {}
    for label, relative in expected.items():
        output = artifacts.get(label) if isinstance(artifacts, dict) else None
        binding = output.get("artifact") if isinstance(output, dict) else None
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != relative
        ):
            raise Stage19ReceiptError(
                f"Stage-19 receipt has a noncanonical {label} binding"
            )
        try:
            payload = (root / relative).read_bytes()
        except OSError as exc:
            raise Stage19ReceiptError(f"cannot read bound Stage-19 {label}") from exc
        if hashlib.sha256(payload).hexdigest() != binding.get("sha256"):
            raise Stage19ReceiptError(
                f"Stage-19 {label} bytes changed after receipt validation"
            )
        payloads[label] = payload
    if receipt_path.read_bytes() != receipt_bytes:
        raise Stage19ReceiptError("Stage-19 receipt changed while reading bound tables")
    return payloads["probability_scores"], payloads["point_scores"]


def _main_under_lock() -> None:
    receipt_path = C.TABLES / "probabilistic_evaluation_v2.json"
    _validate_stage19_receipt(receipt_path)
    probability_bytes, point_bytes = _read_bound_stage19_tables(
        receipt_path, root=ROOT
    )
    probability = pd.read_csv(BytesIO(probability_bytes), float_precision="round_trip")
    point = pd.read_csv(BytesIO(point_bytes), float_precision="round_trip")
    lines = [
        "# Route-A USGS verification synthesis\n",
        "The 2019--2020 period is a previously inspected development evaluation. "
        "All point models use a common sample registry. Event probabilities are "
        "calibrated on 2018 and compared with the bundle-frozen 2006--2018 "
        "seasonal event reference.\n",
        "## Common-key point scores\n",
        "| model | h | n | RMSE | MAE | NSE | KGE | PBIAS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in point.iterrows():
        lines.append(
            f"| {row.model} | {int(row.horizon)} | {int(row.n_common)} | "
            f"{row.RMSE:.3f} | {row.MAE:.3f} | {row.NSE:.3f} | "
            f"{row.KGE:.3f} | {row.PBIAS:+.2f} |"
        )
    lines.extend([
        "",
        "## Probabilistic diagnostics\n",
        "| model | h | PICP | width | 3Q score | calibrated Brier | BSS | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for _, row in probability.iterrows():
        lines.append(
            f"| {row.model} | {int(row.horizon)} | {row.PICP:.3f} | "
            f"{row.MPIW:.2f} | {row.THREE_QUANTILE_SCORE:.3f} | "
            f"{row.Brier_calibrated:.3f} | {row.BrierSkill_calibrated:+.3f} | "
            f"{row.ECE_calibrated:.3f} |"
        )
    lines.extend([
        "",
        "Learned relaxation rates and lag allocations are retained only as internal "
        "latent diagnostics. They are not reported as travel times, variable "
        "importance, causal mechanisms, or physically identified parameters.",
    ])
    atomic_write_bytes(C.REPORTS / "usgs_analysis.md", "\n".join(lines).encode())
    print("\n".join(lines))


def main() -> None:
    # Use the same global Stage16 -> Stage19 order as the Stage19 producer.
    # Holding both shared locks across isolated verification, byte reads,
    # parsing, and publication closes both upstream and Stage19 replacement
    # races without creating a lock-order cycle.
    with advisory_file_lock(C.STAGE16_TRANSACTION_LOCK, exclusive=False):
        with advisory_file_lock(C.STAGE19_TRANSACTION_LOCK, exclusive=False):
            _main_under_lock()


if __name__ == "__main__":
    main()
