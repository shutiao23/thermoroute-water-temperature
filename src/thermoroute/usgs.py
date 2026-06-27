"""Large-sample data acquisition from USGS NWIS + Daymet.

Builds station panels in the *same schema* as the original three-station study
(`DATE, site_id, WTEMP, FLOW, WLEVEL, TEMP, PRCP, WDSP, RHMEAN, DH`) so the
existing ThermoRoute pipeline runs unchanged on dozens of stations.

* NWIS daily values: 00010 water temperature, 00060 discharge, 00065 gage height.
* Daymet single-pixel met at the station coordinates: air temperature
  (mean of tmax/tmin), precipitation, solar radiation (a physical replacement for
  the original, semantically-unverified `DH`), and relative humidity derived from
  vapour pressure. Wind (`WDSP`) is left missing (Daymet has none) and imputed by
  the pipeline; it can be back-filled from gridMET if needed.

All sources are public domain (USGS) / open (Daymet, ORNL DAAC).
"""

from __future__ import annotations

import io
import time
import urllib.request

import numpy as np
import pandas as pd

import dataretrieval.nwis as nwis

NWIS_PARAMS = {"00010": "WTEMP", "00060": "FLOW", "00065": "WLEVEL"}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_sites(state: str, param: str = "00010") -> pd.DataFrame:
    """Return stream ('ST') sites in a state that have daily values for ``param``."""
    out = nwis.what_sites(stateCd=state, parameterCd=param, siteType="ST",
                          hasDataTypeCd="dv")
    df = out[0] if isinstance(out, tuple) else out
    keep = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "alt_va", "huc_cd"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["state"] = state
    return df.dropna(subset=["dec_lat_va", "dec_long_va"])


# --------------------------------------------------------------------------- #
# NWIS daily values
# --------------------------------------------------------------------------- #
def _pick_mean_col(df: pd.DataFrame, pcode: str) -> str | None:
    cands = [c for c in df.columns
             if pcode in c and c.endswith("Mean") and not c.endswith("_cd")]
    return cands[0] if cands else None


def fetch_nwis_daily(site: str, start: str, end: str) -> pd.DataFrame | None:
    """Daily WTEMP/FLOW/WLEVEL for one site, reindexed to a gap-free calendar."""
    try:
        out = nwis.get_record(sites=site, service="dv", start=start, end=end,
                              parameterCd=list(NWIS_PARAMS))
        raw = out[0] if isinstance(out, tuple) else out
    except Exception:
        return None
    if raw is None or len(raw) == 0:
        return None
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    cols = {}
    for pcode, var in NWIS_PARAMS.items():
        col = _pick_mean_col(raw, pcode)
        if col is not None:
            cols[var] = pd.to_numeric(raw[col], errors="coerce")
    if "WTEMP" not in cols or "FLOW" not in cols:
        return None
    daily = pd.DataFrame(cols)
    daily = daily[~daily.index.duplicated(keep="first")]
    full = pd.date_range(start, end, freq="D")
    daily = daily.reindex(full)
    daily.index.name = "DATE"
    return daily


# --------------------------------------------------------------------------- #
# Daymet meteorology (single pixel)
# --------------------------------------------------------------------------- #
def _svp(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (Pa), Tetens."""
    return 611.0 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def fetch_daymet(lat: float, lon: float, start: str, end: str,
                 retries: int = 3) -> pd.DataFrame | None:
    """Daily TEMP/PRCP/DH(=solar radiation)/RHMEAN at a point from Daymet.

    Daymet covers 1980–present and excludes leap day 12-31 in some years; we
    reindex to a full calendar and let the downstream imputer fill the few gaps.
    """
    s_year, e_year = start[:4], end[:4]
    url = (f"https://daymet.ornl.gov/single-pixel/api/data?lat={lat}&lon={lon}"
           f"&vars=tmax,tmin,prcp,srad,vp&start={s_year}-01-01&end={e_year}-12-31")
    raw = None
    for _ in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read().decode()
            break
        except Exception:
            time.sleep(1.0)
    if raw is None:
        return None
    lines = raw.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith("year,")), None)
    if hdr is None:
        return None
    d = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    date = (pd.to_datetime(d["year"].astype(int).astype(str) + "-01-01")
            + pd.to_timedelta(d["yday"].astype(int) - 1, unit="D"))
    tmean = ((d["tmax (deg c)"] + d["tmin (deg c)"]) / 2.0).to_numpy()
    # use plain arrays throughout to avoid pandas index-alignment producing NaN
    rh = 100.0 * d["vp (Pa)"].to_numpy() / _svp(tmean)
    met = pd.DataFrame({
        "TEMP": tmean,
        "PRCP": d["prcp (mm/day)"].to_numpy(),
        "DH": d["srad (W/m^2)"].to_numpy(),          # solar radiation index
        "RHMEAN": np.clip(rh, 0, 100),
    }, index=pd.DatetimeIndex(date, name="DATE"))
    full = pd.date_range(f"{s_year}-01-01", f"{e_year}-12-31", freq="D")
    return met.reindex(full)


# --------------------------------------------------------------------------- #
# gridMET wind speed (point, via NWK THREDDS NetCDF Subset Service)
# --------------------------------------------------------------------------- #
def fetch_gridmet_wind(lat: float, lon: float, start: str, end: str,
                       retries: int = 3) -> pd.Series | None:
    """Daily mean wind speed at a point from gridMET (CONUS, 1979–present).

    Returned as a wind *index* — gridMET NCSS returns packed values, but the
    models z-score WDSP so the absolute scale is irrelevant; we divide by 10 for
    readable magnitudes (≈ m/s).
    """
    url = ("https://thredds.northwestknowledge.net/thredds/ncss/"
           "agg_met_vs_1979_CurrentYear_CONUS.nc?var=daily_mean_wind_speed"
           f"&latitude={lat}&longitude={lon}"
           f"&time_start={start}T00:00:00Z&time_end={end}T00:00:00Z&accept=csv")
    raw = None
    for _ in range(retries):
        try:
            raw = urllib.request.urlopen(url, timeout=60).read().decode()
            break
        except Exception:
            time.sleep(1.0)
    if raw is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(raw))
        wind_col = [c for c in df.columns if "wind" in c.lower()][0]
        idx = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
        s = pd.Series(pd.to_numeric(df[wind_col], errors="coerce").to_numpy() / 10.0,
                      index=idx, name="WDSP")
        return s[~s.index.duplicated()]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Assemble one station into the study schema
# --------------------------------------------------------------------------- #
def build_station(site: str, lat: float, lon: float, site_id: str,
                  start: str, end: str, min_wtemp_cov: float = 0.0,
                  min_flow_cov: float = 0.0) -> tuple[pd.DataFrame | None, dict]:
    nwis_df = fetch_nwis_daily(site, start, end)
    if nwis_df is None:
        return None, {"site": site, "ok": False, "reason": "no NWIS WTEMP+FLOW"}
    wt_cov = nwis_df["WTEMP"].notna().mean()
    fl_cov = nwis_df["FLOW"].notna().mean()
    # reject on coverage BEFORE the (slower) Daymet call to speed up probing
    if wt_cov < min_wtemp_cov or fl_cov < min_flow_cov:
        return None, {"site": site, "ok": False, "reason": "low coverage",
                      "wtemp_cov": round(float(wt_cov), 3),
                      "flow_cov": round(float(fl_cov), 3)}
    met = fetch_daymet(lat, lon, start, end)
    if met is None:
        return None, {"site": site, "ok": False, "reason": "no Daymet"}
    df = nwis_df.join(met, how="left")
    if "WLEVEL" not in df:
        df["WLEVEL"] = np.nan
    wind = fetch_gridmet_wind(lat, lon, start, end)   # gridMET wind index
    df["WDSP"] = wind.reindex(df.index).to_numpy() if wind is not None else np.nan
    df = df.reset_index().rename(columns={"index": "DATE"})
    df.insert(1, "site_id", site_id)
    df = df[["DATE", "site_id", "WTEMP", "FLOW", "WLEVEL",
             "TEMP", "PRCP", "WDSP", "RHMEAN", "DH"]]
    info = {"site": site, "site_id": site_id, "ok": True,
            "wtemp_cov": round(float(wt_cov), 3), "flow_cov": round(float(fl_cov), 3),
            "wlevel_cov": round(float(df["WLEVEL"].notna().mean()), 3),
            "n_days": len(df)}
    return df, info
