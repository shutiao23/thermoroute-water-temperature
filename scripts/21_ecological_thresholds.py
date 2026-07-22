#!/usr/bin/env python3
"""Guarded observed-7DADM threshold description.

Route A predicts daily mean water temperature.  This stage will only compute
7DADM when separately acquired daily-maximum observations and a sourced,
site-specific standards registry exist. It never infers a Gaussian distribution
from three quantiles and never selects stations from evaluation event rates.
"""
# ruff: noqa: E402 -- repository scripts add src to sys.path before local imports.
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thermoroute import config as C
from thermoroute.ecology import (
    OBSERVED_THRESHOLD_COLUMNS,
    evaluate_observed_7dadm,
    load_standard_registry,
)
from thermoroute.repro import atomic_write_bytes


DAILY_MAXIMUM = ROOT / "data_usgs" / "wtemp_daily_max.parquet"
STANDARDS = ROOT / "data_usgs" / "ecological_standards.csv"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_not_performed(output: Path, report: Path, reason: str) -> str:
    empty = pd.DataFrame(columns=OBSERVED_THRESHOLD_COLUMNS)
    atomic_write_bytes(output, empty.to_csv(index=False).encode())
    message = "\n".join([
        "# Observed 7DADM threshold description not performed\n",
        "The Route-A model target is daily mean water temperature. Daily mean "
        "values and model predictions are not observed 7DADM. This descriptive "
        "observed-water-temperature analysis additionally requires independently "
        "sourced daily maxima and a site-specific registry of jurisdiction, "
        "designated use, species/life stage, applicable season, threshold, and "
        "source.",
        f"The stage fails closed: {reason} No observed exceedance classification, "
        "compliance determination, or management-value claim is produced.\n",
        f"Expected daily maxima: `{_display_path(DAILY_MAXIMUM)}`",
        f"Expected standards: `{_display_path(STANDARDS)}`",
    ])
    atomic_write_bytes(report, message.encode())
    print(message)
    return message


def main() -> None:
    output = C.TABLES / "eco_thresholds.csv"
    report = C.REPORTS / "ecological_thresholds.md"
    if not DAILY_MAXIMUM.exists() or not STANDARDS.exists():
        _write_not_performed(output, report, "one or both required inputs are absent.")
        return

    try:
        maxima = pd.read_parquet(DAILY_MAXIMUM)
        standards = load_standard_registry(STANDARDS)
        applied = evaluate_observed_7dadm(maxima, standards)
    except Exception as error:
        _write_not_performed(
            output,
            report,
            f"input validation or threshold application was refused ({error}).",
        )
        raise
    atomic_write_bytes(output, applied.to_csv(index=False).encode())
    comparable = applied.exceedance.notna()
    exceedances = int(applied.loc[comparable, "exceedance"].sum())
    message = "\n".join([
        "# Observed 7DADM threshold description\n",
        f"Computed strict seven-consecutive-day 7DADM for {applied.site_no.nunique()} "
        f"sites and applied {len(standards)} contextual standard records. There are "
        f"{int(comparable.sum())} observed endpoint/context comparisons, of which "
        f"{exceedances} are above the registered threshold.",
        "Applicability is based on the ending date of each seven-day window. Rows "
        "outside a context's declared season or without seven observed daily maxima "
        "receive no exceedance classification.",
        "These are descriptive comparisons of observations with registry entries. "
        "No model prediction is read or classified, and the output is not a legal or "
        "regulatory compliance determination.",
    ])
    atomic_write_bytes(report, message.encode())
    print(message)


if __name__ == "__main__":
    main()
