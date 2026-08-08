#!/usr/bin/env python3
"""Discharge-value ablation on the independent 2021-2023 window.

The external review (Major Comment 7) requires the flow contribution to be
tested by retraining without the discharge channel, not only by synthetic
perturbations.  This script fits a station-agnostic LightGBM per lead on the
development train/validation rows with every feature EXCEPT the FLOW columns
(the same frozen preprocessing, keys, and station weighting as the main
LightGBM), scores the 2021-2023 common keys, and reports the station-first
paired DeltaRMSE against the with-flow LightGBM from the results authority.

Outputs:
  outputs/final/flow_ablation_effects.parquet
  outputs/final/flow_ablation_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import data as D  # noqa: E402
from thermoroute import features as F  # noqa: E402
from thermoroute.baselines import _lgb_fit  # noqa: E402

FINAL = ROOT / "outputs" / "final"
OUT_CONV = ROOT / "outputs" / "conventional"

USGS_VARS = ("WTEMP", "FLOW", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")
MIN_TRAIN_ROWS = 500
MIN_TARGETS = 100

FROZEN_PARAMS = {1: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 3: {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
                 7: {"num_leaves": 15, "min_child_samples": 40, "learning_rate": 0.03}}
BEST_ITER = {1: 800, 3: 575, 7: 440}


def main() -> int:
    registry = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv")
    stations = tuple(sorted(registry.site_no.astype(str).str.zfill(8)))
    C.STATIONS = stations

    train_panel = pd.read_parquet(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")
    train_panel["DATE"] = pd.to_datetime(train_panel["DATE"])
    test_panel = pd.read_parquet(OUT_CONV / "panel_2021_2023.parquet")
    test_panel["DATE"] = pd.to_datetime(test_panel["DATE"])
    alias = dict(zip(registry.legacy_site_id.astype(str),
                     registry.site_no.astype(str).str.zfill(8)))
    train_panel["site_id"] = train_panel["site_id"].astype(str).map(alias)
    for df in (train_panel, test_panel):
        df["site_id"] = df["site_id"].astype(str).str.zfill(8)
    panel = pd.concat([train_panel, test_panel], ignore_index=True)
    panel = panel.drop_duplicates(["DATE", "site_id"], keep="last")
    panel = panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)

    dates = panel["DATE"].to_numpy()
    fit_mask = (dates <= np.datetime64("2017-12-31"))
    imputer = D.Imputer.fit(panel, fit_mask)
    panel_imp = imputer.transform(panel)
    clim = F.HarmonicClimatology.fit(panel, fit_mask, fit_stations=stations)

    keys = pd.read_parquet(FINAL / "forecast_keys.parquet",
                           columns=["site_id", "horizon", "issue_date", "y_true"])
    keys["site_id"] = keys["site_id"].astype(str).str.zfill(8)
    keys["issue_date"] = pd.to_datetime(keys["issue_date"])

    rows: list[dict] = []
    for h in (1, 3, 7):
        tab = F.attach_split(F.build_tabular(
            panel_imp, h, USGS_VARS, clim,
            drop_feature_nans=False, require_observed_target=True,
            include_missingness=True))
        tab.loc[tab.issue_date >= np.datetime64("2021-01-01"), "split"] = "confirm"
        all_cols = F.feature_columns(tab)
        flow_cols = [c for c in all_cols if "FLOW" in c]
        no_flow_cols = [c for c in all_cols if "FLOW" not in c]
        print(f"h{h}: {len(flow_cols)} FLOW columns dropped, "
              f"{len(no_flow_cols)} remaining", flush=True)
        for c in all_cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").fillna(0.0)
        tr = tab[tab.split.isin(["train", "val"])]
        if len(tr) < MIN_TRAIN_ROWS:
            continue
        ytr = tr["y"].to_numpy(float)
        m = _lgb_fit(tr[no_flow_cols], ytr,
                     tr[no_flow_cols].to_numpy(float)[-2000:], ytr[-2000:],
                     "regression", n_est=BEST_ITER[h],
                     params_override=FROZEN_PARAMS[h])
        ev = tab[tab.split.eq("confirm")].merge(
            keys[["site_id", "issue_date"]].drop_duplicates(),
            on=["site_id", "issue_date"], validate="many_to_one")
        if ev.empty:
            continue
        yhat = m.predict(ev[no_flow_cols], num_threads=1)
        frame = pd.DataFrame({
            "site_id": ev.site_id.to_numpy(), "horizon": h,
            "issue_date": ev.issue_date.to_numpy(),
            "y_pred": yhat, "y_true": ev["y"].to_numpy(),
        })
        frame = frame.drop_duplicates(["site_id", "issue_date"])
        rows.append(frame)
        print(f"h{h}: no-flow LightGBM RMSE {np.sqrt(np.mean((yhat-frame.y_true)**2)):.3f}",
              flush=True)

    if not rows:
        raise SystemExit("no predictions produced")
    no_flow = pd.concat(rows, ignore_index=True)
    no_flow.to_parquet(FINAL / "flow_ablation_effects.parquet", index=False)

    with_flow = pd.read_parquet(FINAL / "predictions.parquet")
    with_flow = with_flow[with_flow.model == "LightGBM"][
        ["site_id", "horizon", "issue_date", "y_pred"]].rename(
        columns={"y_pred": "wf"})
    merged = no_flow.merge(with_flow, on=["site_id", "horizon", "issue_date"],
                           validate="one_to_one")

    metric_rows = []
    for (site, h), g in merged.groupby(["site_id", "horizon"]):
        if len(g) < MIN_TARGETS:
            continue
        e_nf = g.y_pred.to_numpy(float) - g.y_true.to_numpy(float)
        e_wf = g.wf.to_numpy(float) - g.y_true.to_numpy(float)
        metric_rows.append({
            "site_id": site, "horizon": int(h), "n": int(len(g)),
            "rmse_no_flow": float(np.sqrt(np.mean(e_nf ** 2))),
            "rmse_with_flow": float(np.sqrt(np.mean(e_wf ** 2))),
            "delta_rmse_noflow_minus_flow": float(np.sqrt(np.mean(e_nf ** 2))
                                                  - np.sqrt(np.mean(e_wf ** 2))),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(FINAL / "flow_ablation_effects.parquet", index=False)

    summary = {
        str(h): {
            "n_stations": int(g.site_id.nunique()),
            "median_delta_rmse_noflow_minus_flow": float(g.delta_rmse_noflow_minus_flow.median()),
            "median_rmse_no_flow": float(g.rmse_no_flow.median()),
            "median_rmse_with_flow": float(g.rmse_with_flow.median()),
            "no_flow_win_frac": float((g.delta_rmse_noflow_minus_flow < 0).mean()),
        }
        for h, g in metrics.groupby("horizon")
    }
    (FINAL / "flow_ablation_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
