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
TEST_START = "2019-01-01"        # blind-test window start
MIN_WTEMP_COV = 0.55              # ≥55% over the full 2006–2020 record
MIN_FLOW_COV = 0.70
MIN_WTEMP_COV_TEST = 0.80         # ≥80% over the 2019–2020 blind-test window
MIN_FLOW_COV_TEST = 0.80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10, help="number of good stations to keep")
    ap.add_argument("--states", nargs="*", default=DEFAULT_STATES)
    ap.add_argument("--max-probe", type=int, default=120, help="max candidate sites to test")
    ap.add_argument("--out", default="panel_usgs.parquet", help="output panel filename")
    ap.add_argument("--min-wtemp-cov", type=float, default=MIN_WTEMP_COV,
                    help="full-period WTEMP coverage threshold")
    ap.add_argument("--min-flow-cov", type=float, default=MIN_FLOW_COV,
                    help="full-period FLOW coverage threshold")
    ap.add_argument("--min-wtemp-cov-test", type=float, default=MIN_WTEMP_COV_TEST,
                    help="blind-test-period WTEMP coverage threshold")
    ap.add_argument("--min-flow-cov-test", type=float, default=MIN_FLOW_COV_TEST,
                    help="blind-test-period FLOW coverage threshold")
    ap.add_argument("--rejected-out", default="rejected_sites.csv",
                    help="filename inside data_usgs/ for the rejection registry")
    ap.add_argument("--meta-out", default="stations_meta.csv",
                    help="filename inside data_usgs/ for the kept-station registry")
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

    kept, report_rows, rejected_rows, n_probed = [], [], [], 0
    for _, row in cand.iterrows():
        if len(kept) >= args.target or n_probed >= args.max_probe:
            break
        n_probed += 1
        site = str(row["site_no"]).zfill(8)
        lat, lon = float(row["dec_lat_va"]), float(row["dec_long_va"])
        sid = f"n{len(kept):02d}"
        df, info = usgs.build_station(
            site, lat, lon, sid, START, END,
            min_wtemp_cov=args.min_wtemp_cov,
            min_flow_cov=args.min_flow_cov,
            min_wtemp_cov_test=args.min_wtemp_cov_test,
            min_flow_cov_test=args.min_flow_cov_test,
            test_start=TEST_START,
        )
        if df is None:
            # record every rejected candidate so the inclusion process is
            # transparent and re-runnable by a third party
            rej = {"site": site,
                   "station_nm": str(row.get("station_nm", "")).strip(),
                   "lat": lat, "lon": lon,
                   "state": row.get("state", ""),
                   "reason": info.get("reason", "unknown"),
                   "wtemp_cov": info.get("wtemp_cov", float("nan")),
                   "flow_cov": info.get("flow_cov", float("nan")),
                   "wtemp_cov_test": info.get("wtemp_cov_test", float("nan")),
                   "flow_cov_test": info.get("flow_cov_test", float("nan"))}
            rejected_rows.append(rej)
            continue
        df.to_csv(OUTDIR / f"{sid}.csv", index=False)
        info["station_nm"] = str(row.get("station_nm", "")).strip()
        info["lat"], info["lon"] = lat, lon
        info["state"] = row.get("state", "")
        kept.append(df)
        report_rows.append(info)
        print(f"[keep {len(kept):2d}/{args.target}] {sid} {site} "
              f"wtemp_cov={info['wtemp_cov']} flow_cov={info['flow_cov']} "
              f"wt_test={info['wtemp_cov_test']} fl_test={info['flow_cov_test']} "
              f"{info['station_nm'][:40]}", flush=True)

    # ALWAYS write the rejection registry, even if nothing was kept — this is
    # the auditable record of how the inclusion criteria filtered candidates.
    if rejected_rows:
        pd.DataFrame(rejected_rows).to_csv(OUTDIR / args.rejected_out, index=False)
        print(f"[rejected] {len(rejected_rows)} candidates -> "
              f"{OUTDIR / args.rejected_out}", flush=True)

    if not kept:
        print("no stations met the coverage thresholds — widen states/--max-probe")
        return

    panel = pd.concat(kept, ignore_index=True)
    panel.to_parquet(OUTDIR / args.out)
    rep = pd.DataFrame(report_rows)
    rep.to_csv(OUTDIR / args.meta_out, index=False)

    # report
    L = [f"# USGS large-sample acquisition ({len(kept)} stations)\n",
         f"_Window {START}…{END}. Probed {n_probed} candidates in "
         f"{time.time()-t0:.0f}s. Schema matches the original study._\n",
         "## Inclusion criteria\n",
         f"- Full-record WTEMP coverage ≥ {args.min_wtemp_cov:.2f}",
         f"- Full-record FLOW coverage ≥ {args.min_flow_cov:.2f}",
         f"- Blind-test-window ({TEST_START}–{END}) WTEMP coverage ≥ {args.min_wtemp_cov_test:.2f}",
         f"- Blind-test-window FLOW coverage ≥ {args.min_flow_cov_test:.2f}\n",
         "These thresholds ensure that *every* accepted station can both train "
         "the model on the pre-2019 record and contribute observations to the "
         "2019–2020 blind-test evaluation. Every probed candidate (kept or "
         f"rejected) is recorded in `data_usgs/{args.rejected_out}` and "
         f"`data_usgs/{args.meta_out}` so the inclusion process is auditable.\n",
         "## Kept stations\n",
         "| site_id | USGS | state | name | WT cov | FL cov | WT cov 2019+ | FL cov 2019+ | WLEVEL cov |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in report_rows:
        L.append(f"| {r['site_id']} | {r['site']} | {r['state']} | "
                 f"{r['station_nm'][:34]} | {r['wtemp_cov']} | {r['flow_cov']} | "
                 f"{r.get('wtemp_cov_test', '-')} | {r.get('flow_cov_test', '-')} | "
                 f"{r['wlevel_cov']} |")
    L += ["", f"## Rejection summary ({len(rejected_rows)} candidates probed and rejected)\n"]
    if rejected_rows:
        rejdf = pd.DataFrame(rejected_rows)
        L.append("| reason | count |"); L.append("|---|---|")
        for reason, n in rejdf["reason"].value_counts().items():
            L.append(f"| {reason} | {n} |")
        L.append(f"\nFull per-site detail: `data_usgs/{args.rejected_out}`.")
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
