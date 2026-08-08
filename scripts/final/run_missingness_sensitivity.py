#!/usr/bin/env python3
"""Missingness-stratified evaluation on the 2021-2023 window.

The external review (Major Comment 12) requires that model-vs-anchor
conclusions are not dominated by keys whose recent water-temperature history is
mostly imputed.  This script stratifies the station-first paired DeltaRMSE
(ThermoRoute - damped persistence) by the recent-history observed fraction
recorded in the authority key registry:

    strata: >=0.95, 0.75-0.95, 0.50-0.75, <0.50  (7d / 14d / 32d windows)

plus an issue-date-observed-only cell.  All metrics are station-first with the
standard 100-key reportability rule; the summary reports the median station
DeltaRMSE per stratum and the station/key counts, so a reader can see whether
the learned-gain conclusion depends on heavily imputed keys.

Output: outputs/final/missingness_sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs" / "final"

MIN_TARGETS = 100


def main() -> int:
    keys = pd.read_parquet(FINAL / "forecast_keys.parquet",
                           columns=["site_id", "issue_date", "horizon",
                                    "fraction_observed_wtemp_7d",
                                    "fraction_observed_wtemp_14d",
                                    "fraction_observed_wtemp_32d"])
    keys["issue_date"] = pd.to_datetime(keys["issue_date"])
    keys["site_id"] = keys["site_id"].astype(str).str.zfill(8)
    pred = pd.read_parquet(FINAL / "predictions.parquet")
    pred["issue_date"] = pd.to_datetime(pred["issue_date"])
    pred["site_id"] = pred["site_id"].astype(str).str.zfill(8)
    pred = pred[pred.model.isin(["ThermoRoute", "DampedPersistence"])]
    pred = pred.merge(keys, on=["site_id", "issue_date", "horizon"], how="left")

    def stratum(series: pd.Series) -> str:
        out = []
        for value in series:
            if not np.isfinite(value):
                out.append("missing_frac_na")
            elif value >= 0.95:
                out.append("frac_ge_0.95")
            elif value >= 0.75:
                out.append("frac_0.75_0.95")
            elif value >= 0.50:
                out.append("frac_0.50_0.75")
            else:
                out.append("frac_lt_0.50")
        return out

    summary: dict = {}
    cand = pred[pred.model == "ThermoRoute"][
        ["site_id", "horizon", "issue_date", "y_true", "y_pred",
         "fraction_observed_wtemp_7d", "fraction_observed_wtemp_14d",
         "fraction_observed_wtemp_32d"]] \
        .rename(columns={"y_pred": "cand"})
    ref = pred[pred.model == "DampedPersistence"][
        ["site_id", "horizon", "issue_date", "y_pred"]] \
        .rename(columns={"y_pred": "ref"})
    both = cand.merge(ref, on=["site_id", "horizon", "issue_date"],
                      validate="one_to_one")
    for window in ("7d", "14d", "32d"):
        column = f"fraction_observed_wtemp_{window}"
        both[f"stratum_{window}"] = stratum(both[column])
        block: dict = {}
        for stratum_name, g in both.groupby("stratum_" + window):
            rows = []
            for (site, horizon), cell in g.groupby(["site_id", "horizon"]):
                if len(cell) < MIN_TARGETS:
                    continue
                e_c = cell["cand"].to_numpy(float) - cell["y_true"].to_numpy(float)
                e_r = cell["ref"].to_numpy(float) - cell["y_true"].to_numpy(float)
                rows.append({
                    "site_id": site, "horizon": horizon, "n": int(len(cell)),
                    "delta_rmse": float(np.sqrt(np.mean(e_c ** 2))
                                        - np.sqrt(np.mean(e_r ** 2))),
                })
            cell_df = pd.DataFrame(rows)
            block[stratum_name] = {
                "n_stations": int(cell_df.site_id.nunique()) if len(cell_df) else 0,
                "n_keys": int(len(g)),
                "median_delta_rmse": float(cell_df.delta_rmse.median())
                if len(cell_df) else None,
            }
        summary[window] = block
    (FINAL / "missingness_sensitivity.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
