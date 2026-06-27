#!/usr/bin/env python3
"""Build a large-sample station set from USGS NWIS + Daymet, in the study schema.

Discovers stream sites with daily water-temperature records, downloads
WTEMP/FLOW/WLEVEL + Daymet meteorology, keeps sites with adequate coverage over
the study window, and writes per-site CSVs + a combined panel + a report.

Usage:
    PYTHONPATH=src python3 scripts/data_usgs/build_usgs_stations.py --target 10
    PYTHONPATH=src python3 scripts/data_usgs/build_usgs_stations.py --target 40 \
        --states CO OR WA PA NY MN WI CA ID MT VT NH

The result drops straight into the ThermoRoute pipeline: point its data loader at
`data_usgs/panel_usgs.parquet` (same columns as the original three-station panel).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from thermoroute import usgs

OUTDIR = ROOT / "data_usgs"
OUTDIR.mkdir(exist_ok=True)
REPORT = ROOT / "outputs" / "reports" / "usgs_acquisition.md"

DEFAULT_STATES = ["CO", "OR", "WA", "PA", "NY", "MN", "WI", "CA", "ID", "MT"]
START, END = "2006-01-01", "2020-12-31"
MIN_WTEMP_COV = 0.55          # ≥55% of the 15-yr daily window observed
MIN_FLOW_COV = 0.70


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10, help="number of good stations to keep")
    ap.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    ap.add_argument("--max-probe", type=int, default=120, help="max candidate sites to test")
    ap.add_argument("--out", default="panel_usgs.parquet", help="output panel filename")
    args = ap.parse_args()

    t0 = time.time()
    # 1. discover candidates across states
    cands = []
    for st in args.states:
        try:
            d = usgs.discover_sites(st)
            cands.append(d)
            print(f"[discover] {st}: {len(d)} candidate sites", flush=True)
        except Exception as e:
            print(f"[discover] {st} failed: {e!r}", flush=True)
    cand = pd.concat(cands, ignore_index=True).drop_duplicates("site_no")
    # deterministic shuffle so probing order is reproducible
    cand = cand.sample(frac=1.0, random_state=0).reset_index(drop=True)

    kept, report_rows, n_probed = [], [], 0
    for _, row in cand.iterrows():
        if len(kept) >= args.target or n_probed >= args.max_probe:
            break
        n_probed += 1
        site = str(row["site_no"]).zfill(8)
        lat, lon = float(row["dec_lat_va"]), float(row["dec_long_va"])
        sid = f"n{len(kept):02d}"
        df, info = usgs.build_station(site, lat, lon, sid, START, END,
                                      min_wtemp_cov=MIN_WTEMP_COV,
                                      min_flow_cov=MIN_FLOW_COV)
        if df is None:
            continue
        df.to_csv(OUTDIR / f"{sid}.csv", index=False)
        info["station_nm"] = str(row.get("station_nm", "")).strip()
        info["lat"], info["lon"] = lat, lon
        info["state"] = row.get("state", "")
        kept.append(df)
        report_rows.append(info)
        print(f"[keep {len(kept):2d}/{args.target}] {sid} {site} "
              f"wtemp_cov={info['wtemp_cov']} flow_cov={info['flow_cov']} "
              f"{info['station_nm'][:40]}", flush=True)

    if not kept:
        print("no stations met the coverage thresholds — widen states/--max-probe")
        return

    panel = pd.concat(kept, ignore_index=True)
    panel.to_parquet(OUTDIR / args.out)
    rep = pd.DataFrame(report_rows)
    rep.to_csv(OUTDIR / "stations_meta.csv", index=False)

    # report
    L = [f"# USGS large-sample acquisition ({len(kept)} stations)\n",
         f"_Window {START}…{END}. Probed {n_probed} candidates in "
         f"{time.time()-t0:.0f}s. Schema matches the original study._\n",
         "| site_id | USGS | state | name | WTEMP cov | FLOW cov | WLEVEL cov |",
         "|---|---|---|---|---|---|---|"]
    for r in report_rows:
        L.append(f"| {r['site_id']} | {r['site']} | {r['state']} | "
                 f"{r['station_nm'][:34]} | {r['wtemp_cov']} | {r['flow_cov']} | "
                 f"{r['wlevel_cov']} |")
    L += ["",
          "Meteorology from Daymet single-pixel (TEMP=mean of tmax/tmin, PRCP, "
          "DH=solar radiation W/m², RHMEAN from vapour pressure). WDSP not in "
          "Daymet — left missing (imputed) and can be added from gridMET.",
          "", f"Combined panel: `data_usgs/panel_usgs.parquet` "
          f"({len(panel)} rows, {len(kept)} sites)."]
    REPORT.write_text("\n".join(L))
    print(f"\nDONE: {len(kept)} stations, {len(panel)} rows -> "
          f"{OUTDIR/'panel_usgs.parquet'}\nreport -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
