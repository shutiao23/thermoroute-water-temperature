#!/usr/bin/env python3
"""Route-A temporal conformal sensitivity with delayed feedback.

Compares ordinary split-CQR, seven-day block-CQR, and delayed-feedback ACI.  All
results are empirical diagnostics on a previously inspected development period;
no exchangeability or conditional-coverage guarantee is claimed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import results as R
from thermoroute.adaptive import delayed_aci
from thermoroute.conformal import block_cqr_offsets, conformal_quantile
from thermoroute.evidence import FrozenPanelSpec
from thermoroute.probability import ensemble_prediction_frame
from thermoroute.repro import atomic_write_bytes
from thermoroute.spatial import huc2_cluster_map, load_station_registry


PREDICTIONS = C.PREDICTIONS / "usgs_predictions_v2.parquet"
PANEL = ROOT / "data_usgs" / "panel_usgs_120v2.parquet"
STATION_REGISTRY = ROOT / "data_usgs" / "station_registry_v1.csv"
ALPHA = 0.10
GAMMAS = (0.005, 0.02, 0.05)


def _interval_score(y, lower, upper, alpha=ALPHA):
    width = upper - lower
    return width + np.where(y < lower, 2 / alpha * (lower - y), 0) + np.where(
        y > upper, 2 / alpha * (y - upper), 0
    )


def main() -> None:
    predictions = R.load_route_a_predictions(
        PREDICTIONS,
        root=ROOT,
        panel_path=PANEL,
        registry_path=STATION_REGISTRY,
    )
    ensemble = ensemble_prediction_frame(predictions, "ThermoRoute")
    ensemble["issue_date"] = pd.to_datetime(ensemble.issue_date)
    ensemble["target_date"] = pd.to_datetime(ensemble.target_date)
    calibration = ensemble[ensemble.split == "calib"].copy()
    evaluation = ensemble[ensemble.split == "test"].copy()
    calibration = calibration[
        calibration.q05.notna() & calibration.q95.notna()
        & (calibration.target_date <= pd.Timestamp(C.SPLIT.calib[1]))
    ]
    evaluation = evaluation[evaluation.q05.notna() & evaluation.q95.notna()]

    panel = FrozenPanelSpec.load().load_panel(stable_site_ids=True)
    panel["DATE"] = pd.to_datetime(panel.DATE)
    masks = D.split_masks(panel.DATE)
    train = panel.loc[masks.train]
    warm_threshold = {
        site: float(group.WTEMP.quantile(C.EXCEEDANCE_QUANTILE))
        for site, group in train.groupby("site_id")
    }
    huc = huc2_cluster_map(load_station_registry(STATION_REGISTRY))
    block_offsets = block_cqr_offsets(calibration, alpha=ALPHA, block_days=7)

    records = []
    for (site, horizon), test_group in evaluation.groupby(["site_id", "horizon"]):
        cal_group = calibration[
            (calibration.site_id == site) & (calibration.horizon == horizon)
        ]
        if len(cal_group) < 30 or len(test_group) < 30:
            continue
        scores = np.maximum(
            cal_group.q05 - cal_group.y_true, cal_group.y_true - cal_group.q95
        ).to_numpy(float)
        split_offset = conformal_quantile(scores, ALPHA)
        block_offset = block_offsets[(site, horizon)]
        test_group = test_group.sort_values("issue_date").copy()
        y = test_group.y_true.to_numpy(float)
        split_lower = test_group.q05.to_numpy(float) - split_offset
        split_upper = test_group.q95.to_numpy(float) + split_offset
        block_lower = test_group.q05.to_numpy(float) - block_offset
        block_upper = test_group.q95.to_numpy(float) + block_offset

        base = pd.DataFrame({
            "site_id": site,
            "horizon": int(horizon),
            "huc2": huc.get(site, "unmapped"),
            "issue_date": test_group.issue_date.to_numpy(),
            "target_date": test_group.target_date.to_numpy(),
            "warm": y >= warm_threshold.get(site, np.inf),
            "split_covered": (y >= split_lower) & (y <= split_upper),
            "split_width": split_upper - split_lower,
            "split_interval_score": _interval_score(y, split_lower, split_upper),
            "block_covered": (y >= block_lower) & (y <= block_upper),
            "block_width": block_upper - block_lower,
            "block_interval_score": _interval_score(y, block_lower, block_upper),
        })
        for gamma in GAMMAS:
            adaptive = delayed_aci(scores, test_group, alpha=ALPHA, gamma=gamma)
            tag = str(gamma).replace(".", "p")
            base[f"aci_{tag}_covered"] = adaptive.aci_covered.to_numpy(bool)
            base[f"aci_{tag}_width"] = adaptive.aci_width.to_numpy(float)
            base[f"aci_{tag}_interval_score"] = adaptive.aci_interval_score.to_numpy(float)
            base[f"aci_{tag}_feedback_count"] = adaptive.feedback_count.to_numpy(int)
        records.append(base)
    result = pd.concat(records, ignore_index=True)
    atomic_write_bytes(C.TABLES / "aci_coverage.csv", result.to_csv(index=False).encode())

    methods = [
        ("split-CQR", "split"),
        ("7-day block-CQR", "block"),
        *[(f"delayed ACI gamma={gamma}", f"aci_{str(gamma).replace('.', 'p')}")
          for gamma in GAMMAS],
    ]
    lines = [
        "# Temporal conformal sensitivity\n",
        "All target feedback is delayed until `target_date`; a 7-day forecast can "
        "therefore not update ACI for the next seven issue days. Results are empirical "
        "development-period diagnostics, not finite-sample guarantees.\n",
        "| method | slice | n | coverage | width | interval score |",
        "|---|---|---|---|---|---|",
    ]
    slices = [("overall", result), ("warm train-q90 tail", result[result.warm])]
    slices.extend((f"lead {h} d", result[result.horizon == h]) for h in C.HORIZONS)
    for method, tag in methods:
        for label, frame in slices:
            lines.append(
                f"| {method} | {label} | {len(frame)} | "
                f"{frame[f'{tag}_covered'].mean():.3f} | "
                f"{frame[f'{tag}_width'].mean():.2f} | "
                f"{frame[f'{tag}_interval_score'].mean():.2f} |"
            )
    lines.extend([
        "",
        "Cross-HUC2 dispersion is descriptive: each method's per-HUC2 coverage "
        "distribution is retained in the row-level CSV for clustered analysis. ACI "
        "is not described as cost-free; width and interval score are reported next "
        "to coverage for every gamma.",
    ])
    atomic_write_bytes(C.REPORTS / "adaptive_conformal.md", "\n".join(lines).encode())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
