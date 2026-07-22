#!/usr/bin/env python3
"""Route-A synthesis of already generated USGS verification artifacts.

Mechanistic claims based on latent kappa or router weights were retired: neither
quantity is identifiable or causal.  Probability calibration and hypothetical
decision sensitivity are produced by stages 19 and 18 respectively.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thermoroute import config as C
from thermoroute.repro import atomic_write_bytes


def main() -> None:
    probabilistic_path = C.TABLES / "probabilistic_scores.csv"
    point_path = C.TABLES / "multi_metric.csv"
    if not probabilistic_path.exists() or not point_path.exists():
        raise FileNotFoundError(
            "run scripts/19_probabilistic.py before the Route-A synthesis"
        )
    probability = pd.read_csv(probabilistic_path)
    point = pd.read_csv(point_path)
    lines = [
        "# Route-A USGS verification synthesis\n",
        "The 2019--2020 period is a previously inspected development evaluation. "
        "All point models use a common sample registry. Event probabilities are "
        "calibrated on 2018 and compared with a training-fitted seasonal climatology.\n",
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


if __name__ == "__main__":
    main()
