#!/usr/bin/env python3
"""Parallel per-station air2stream (a8) on the 2021-2023 held-out keys.

Optimized: fits are parallelized across stations (multiprocessing) and only
the held-out forecast keys are rolled forward (no full-period loop).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402
from thermoroute import air2stream as A2S  # noqa: E402
from thermoroute import features as F  # noqa: E402

OUT = ROOT / "outputs" / "conventional"


def _fit_station(args):
    st, sub = args
    Ta = sub["TEMP"].to_numpy(float)
    Q = sub["FLOW"].to_numpy(float)
    T = sub["WTEMP"].to_numpy(float)
    doy = sub["DATE"].dt.dayofyear.to_numpy()
    obs = ~np.isnan(T)
    fit_obj = A2S.fit(Ta, Q, T, doy, variant="a8", obs=obs)
    return st, fit_obj


def main() -> None:
    train_panel = pd.read_parquet(ROOT / "data_usgs" / "panel_usgs_120v2.parquet")
    train_panel["DATE"] = pd.to_datetime(train_panel["DATE"])
    test_panel = pd.read_parquet(OUT / "panel_2021_2023.parquet")
    test_panel["DATE"] = pd.to_datetime(test_panel["DATE"])
    reg = pd.read_csv(ROOT / "data_usgs" / "station_registry_v1.csv",
                      usecols=["site_no", "legacy_site_id"])
    alias = dict(zip(reg["legacy_site_id"].astype(str),
                     reg["site_no"].astype(str).str.zfill(8)))
    train_panel["site_id"] = train_panel["site_id"].astype(str).map(alias)
    for df in (train_panel, test_panel):
        df["site_id"] = df["site_id"].astype(str).str.zfill(8)
    train_panel = train_panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)

    stations = sorted(test_panel.site_id.unique())
    C.STATIONS = tuple(stations)
    print(f"fitting {len(stations)} stations", flush=True)

    import multiprocessing as mp
    sub_frames = [(st, train_panel[train_panel.site_id == st]) for st in stations]
    with mp.Pool(min(16, len(stations))) as pool:
        fits = dict(pool.imap_unordered(_fit_station, sub_frames))

    test_panel = test_panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)
    clim_air = F.HarmonicClimatology.fit(
        train_panel, np.ones(len(train_panel), dtype=bool), target="TEMP",
        fit_stations=stations)

    ref = pd.read_parquet(OUT / "predictions_2021_2023.parquet",
                          columns=["site_id", "horizon", "issue_date", "y_true"])
    ref["site_id"] = ref["site_id"].astype(str).str.zfill(8)
    ref["issue_date"] = pd.to_datetime(ref["issue_date"])
    keys = ref[["site_id", "horizon", "issue_date", "y_true"]].drop_duplicates(
        ["site_id", "horizon", "issue_date"])

    test_panel["doy"] = test_panel["DATE"].dt.dayofyear
    rows = []
    for st, g in keys.groupby("site_id"):
        if st not in fits or g.empty:
            continue
        fit_obj = fits[st]
        sub = test_panel[test_panel.site_id == st].set_index("DATE")
        for h in (1, 3, 7):
            kg = g[g.horizon == h]
            for _, k in kg.iterrows():
                issue = k["issue_date"]
                if issue not in sub.index:
                    continue
                row = sub.loc[issue]
                T0, Q0, doy0 = (float(row["WTEMP"]), float(row["FLOW"]),
                                int(row["doy"]))
                if not np.isfinite(T0) or not np.isfinite(Q0) or Q0 <= 0:
                    continue
                doy_f = [int((issue + pd.Timedelta(days=j + 1)).dayofyear)
                         for j in range(h)]
                ta_step1 = float(row["TEMP"]) if np.isfinite(float(row["TEMP"])) \
                    else float(clim_air.predict(st, np.array([doy0]))[0])
                Ta_f = np.array([ta_step1] + [float(clim_air.predict(
                    st, np.array([d]))[0]) for d in doy_f[1:]])
                yhat = A2S.forecast_horizon(fit_obj, T0, Q0, doy0, Ta_f,
                                            np.array(doy_f, dtype=float))
                rows.append({"site_id": st, "horizon": h, "issue_date": issue,
                             "y_pred": yhat, "y_true": k["y_true"]})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "air2stream_2021_2023.parquet")
    per = out.groupby("horizon").apply(
        lambda g: np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)), include_groups=False)
    med = out.groupby(["site_id", "horizon"]).apply(
        lambda g: np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)),
        include_groups=False).groupby("horizon").median()
    damped = pd.read_csv(OUT / "station_metrics_2021_2023.csv")
    d_med = damped[damped.model == "DampedPersistence"].groupby("horizon")["rmse"].median()
    print("pooled RMSE:", per.round(3).to_dict(), flush=True)
    print("station-median RMSE:", med.round(3).to_dict(), flush=True)
    print("damped station-median RMSE:", d_med.round(3).to_dict(), flush=True)
    print("skill vs damped:", (1 - med / d_med).round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
