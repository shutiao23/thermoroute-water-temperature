#!/usr/bin/env python3
"""Guarded 7DADM audit.

Route A predicts daily mean water temperature.  This stage will only compute
7DADM when separately acquired daily-maximum observations and an authoritative,
site-specific standards registry exist.  It never infers a Gaussian distribution
from three quantiles and never selects stations from evaluation event rates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from thermoroute import config as C
from thermoroute.ecology import compute_7dadm, load_standard_registry
from thermoroute.repro import atomic_write_bytes


DAILY_MAXIMUM = ROOT / "data_usgs" / "wtemp_daily_max.parquet"
STANDARDS = ROOT / "data_usgs" / "ecological_standards.csv"


def main() -> None:
    output = C.TABLES / "eco_thresholds.csv"
    report = C.REPORTS / "ecological_thresholds.md"
    if not DAILY_MAXIMUM.exists() or not STANDARDS.exists():
        empty = pd.DataFrame(columns=[
            "site_no", "DATE", "WTEMP_MAX", "SEVEN_DADM", "SEVEN_DADM_N",
            "threshold_c", "applicable", "exceedance",
        ])
        atomic_write_bytes(output, empty.to_csv(index=False).encode())
        message = "\n".join([
            "# Regulatory/ecological threshold analysis not performed\n",
            "The Route-A model target is daily mean water temperature. Daily mean "
            "exceedance is not EPA 7DADM. A defensible 7DADM analysis additionally "
            "requires independently sourced daily maxima and a site-specific registry "
            "of jurisdiction, designated use, species/life stage, applicable season, "
            "threshold, and source. One or both frozen inputs are absent, so no "
            "regulatory score or management-value claim is produced.\n",
            f"Expected daily maxima: `{DAILY_MAXIMUM.relative_to(ROOT)}`",
            f"Expected standards: `{STANDARDS.relative_to(ROOT)}`",
        ])
        atomic_write_bytes(report, message.encode())
        print(message)
        return

    maxima = pd.read_parquet(DAILY_MAXIMUM)
    standards = load_standard_registry(STANDARDS)
    seven = compute_7dadm(maxima)
    # The correct observed 7DADM series is retained for data audit. Route A has no
    # daily-maximum forecast head, so it is not scored as a forecast here.
    atomic_write_bytes(output, seven.to_csv(index=False).encode())
    message = "\n".join([
        "# Observed 7DADM audit\n",
        f"Computed strict seven-consecutive-day 7DADM for {seven.site_no.nunique()} sites. "
        f"Validated {len(standards)} contextual standard records. Route A predicts "
        "daily means, so no forecast-skill or regulatory-compliance claim is made.",
    ])
    atomic_write_bytes(report, message.encode())
    print(message)


if __name__ == "__main__":
    main()
