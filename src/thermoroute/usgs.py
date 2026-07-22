"""Large-sample data acquisition from USGS NWIS + Daymet.

Builds station panels in the *same schema* as the original three-station study
(`DATE, site_id, WTEMP, FLOW, WLEVEL, TEMP, PRCP, WDSP, RHMEAN, DH`) so the
existing ThermoRoute pipeline runs unchanged on dozens of stations.

* NWIS daily values: 00010 water temperature, 00060 discharge, 00065 gage height.
* Daymet single-pixel met at the station coordinates: a tmax/tmin temperature
  proxy, precipitation, daylight-period mean solar flux (stored under the legacy
  feature name ``DH``), and a vapour-pressure/Tetens humidity proxy.  These proxy
  names are not process measurements. Wind (`WDSP`) is obtained from gridMET.

All sources are public domain (USGS) / open (Daymet, ORNL DAAC).
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .provenance import SnapshotStore

NWIS_PARAMS = {"00010": "WTEMP", "00060": "FLOW", "00065": "WLEVEL"}
CONFIRMATORY_OUTCOME_COLUMNS = (
    "site_no",
    "DATE",
    *(
        column
        for variable in NWIS_PARAMS.values()
        for column in (
            variable,
            f"{variable}_qualifier",
            f"{variable}_series_id",
            f"{variable}_qualifier_column",
            f"{variable}_series_conflict",
            f"{variable}_conflicting_series_count",
            f"{variable}_conflicting_series_ids",
            f"{variable}_conflicting_series_qualifiers",
            f"{variable}_conflicting_series_provenance",
            f"{variable}_value_status",
        )
    ),
)
CONFIRMATORY_NWIS_PROVIDER = "usgs-nwis-confirmatory-dv"
GRIDMET_WIND_DATASET = "agg_met_vs_1979_CurrentYear_CONUS.nc"
GRIDMET_WIND_SCALE_FACTOR = 0.1
GRIDMET_WIND_ADD_OFFSET = 0.0


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _parse_nwis_rdb(payload: bytes) -> pd.DataFrame:
    """Parse a raw NWIS RDB response while retaining site_no as text."""
    text = payload.decode("utf-8", errors="strict")
    rows = [line for line in text.splitlines() if line and not line.startswith("#")]
    if not rows:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(rows)), sep="\t", dtype=str)
    # RDB places a field-width/type declaration (for example ``5s``) directly
    # below the header.  It is metadata, not an observation.
    if len(df) and all(
        pd.isna(value) or re.fullmatch(r"\d+[a-z]", str(value).strip())
        for value in df.iloc[0]
    ):
        df = df.iloc[1:].reset_index(drop=True)
    return df


def discover_sites(
    state: str,
    param: str = "00010",
    snapshot_store: SnapshotStore | None = None,
) -> pd.DataFrame:
    """Return stream ('ST') sites in a state that have daily values for ``param``."""
    if snapshot_store is None:
        import dataretrieval.nwis as nwis

        out = nwis.what_sites(stateCd=state, parameterCd=param, siteType="ST",
                              hasDataTypeCd="dv")
        df = out[0] if isinstance(out, tuple) else out
    else:
        url = "https://waterservices.usgs.gov/nwis/site/?" + urlencode({
            "format": "rdb",
            "stateCd": state,
            "parameterCd": param,
            "siteType": "ST",
            "hasDataTypeCd": "dv",
            "siteStatus": "all",
        })
        payload, _ = snapshot_store.fetch(provider="usgs-nwis-site", url=url)
        df = _parse_nwis_rdb(payload)
        for col in ("dec_lat_va", "dec_long_va", "alt_va"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "alt_va", "huc_cd"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["state"] = state
    return df.dropna(subset=["dec_lat_va", "dec_long_va"])


# --------------------------------------------------------------------------- #
# NWIS daily values
# --------------------------------------------------------------------------- #
def _mean_columns(df: pd.DataFrame, pcode: str) -> list[str]:
    """Return every daily-mean value column for one parameter code.

    NWIS may expose more than one time-series identifier for the same parameter.
    That is harmless for exploratory acquisition only if the caller makes an
    explicit choice; a confirmatory parser must instead reject the ambiguity.
    """
    return [
        str(column)
        for column in df.columns
        if pcode in str(column)
        and (str(column).endswith("Mean") or str(column).endswith("_00003"))
        and not str(column).endswith("_cd")
    ]


def _pick_mean_col(
    df: pd.DataFrame,
    pcode: str,
    *,
    reject_ambiguous: bool = False,
) -> str | None:
    candidates = _mean_columns(df, pcode)
    if reject_ambiguous and len(candidates) > 1:
        raise ValueError(
            f"NWIS response has multiple daily-mean columns for {pcode}: "
            f"{sorted(candidates)}"
        )
    return candidates[0] if candidates else None


def nwis_confirmatory_series_registry(payload: bytes) -> dict[str, list[dict[str, str | None]]]:
    """Return every exact daily-mean value/qualifier column pair in an RDB response."""
    raw = _parse_nwis_rdb(payload)
    registry: dict[str, list[dict[str, str | None]]] = {}
    for parameter_code, variable in NWIS_PARAMS.items():
        registry[variable] = [
            {
                "parameter_code": parameter_code,
                "value_column": column,
                "qualifier_column": (
                    f"{column}_cd" if f"{column}_cd" in raw.columns else None
                ),
            }
            for column in sorted(_mean_columns(raw, parameter_code))
        ]
    return registry


def _validate_confirmatory_nwis_row_identity(
    raw: pd.DataFrame,
    *,
    site_no: str,
) -> None:
    """Require provider, station and date identity on every non-empty RDB row."""
    required = {"agency_cd", "site_no", "datetime"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            f"non-empty NWIS response lacks identity columns: {missing}"
        )
    if raw.empty:
        return
    cleaned: dict[str, pd.Series] = {}
    for column in sorted(required):
        values = raw[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(
                f"non-empty NWIS response has an empty {column} identity"
            )
        cleaned[column] = values
    agencies = set(cleaned["agency_cd"].astype(str))
    if agencies != {"USGS"}:
        raise ValueError(
            f"NWIS response agency registry is not exactly USGS: {sorted(agencies)}"
        )
    returned_sites = set(cleaned["site_no"].astype(str))
    if returned_sites != {site_no}:
        raise ValueError(
            f"NWIS response site registry {sorted(returned_sites)} != {site_no}"
        )


def parse_nwis_confirmatory_daily(
    payload: bytes,
    *,
    site_no: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Parse a frozen NWIS RDB response into the Route-A daily-value registry.

    The parser retains every requested calendar day and every requested field.
    Missing parameters remain NaN; they never cause a site to be replaced or a
    response to be discarded.  Only the daily-mean statistic (00003/``Mean``)
    is admitted.  This pure function is used both by acquisition and by the
    post-opening verifier, so a derived panel cannot silently disagree with its
    immutable raw response.
    """
    site_no = str(site_no).strip()
    if not site_no:
        raise ValueError("confirmatory NWIS parser requires a stable site_no")
    try:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid confirmatory NWIS date interval") from exc
    if first > last:
        raise ValueError("confirmatory NWIS date interval is reversed")
    raw = _parse_nwis_rdb(payload)
    full = pd.date_range(first, last, freq="D", name="DATE")
    output = pd.DataFrame(index=full)
    for variable in NWIS_PARAMS.values():
        output[variable] = np.nan
        # Preserve NWIS approval/provisional codes (for example A or P).  They
        # are evidence about the status of a value, not model inputs.
        output[f"{variable}_qualifier"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
        output[f"{variable}_series_id"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
        output[f"{variable}_qualifier_column"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
        output[f"{variable}_series_conflict"] = False
        output[f"{variable}_conflicting_series_count"] = 0
        output[f"{variable}_value_status"] = "MISSING_NO_FINITE_SERIES"
        output[f"{variable}_conflicting_series_ids"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
        output[f"{variable}_conflicting_series_qualifiers"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
        output[f"{variable}_conflicting_series_provenance"] = pd.Series(
            pd.NA, index=output.index, dtype="string"
        )
    columns = list(CONFIRMATORY_OUTCOME_COLUMNS)
    if len(raw.columns):
        _validate_confirmatory_nwis_row_identity(raw, site_no=site_no)
    if raw.empty:
        return output.reset_index().assign(site_no=site_no)[columns]
    dates = pd.to_datetime(raw["datetime"], errors="coerce")
    if dates.isna().any():
        raise ValueError("NWIS response contains an invalid daily date")
    dates = dates.dt.tz_localize(None).dt.normalize()
    if ((dates < first.normalize()) | (dates > last.normalize())).any():
        raise ValueError("NWIS response contains a date outside the frozen request")
    if dates.duplicated().any():
        raise ValueError("NWIS response duplicates a daily site/date row")
    raw = raw.copy()
    raw.index = pd.DatetimeIndex(dates)
    for parameter_code, variable in NWIS_PARAMS.items():
        candidates = sorted(_mean_columns(raw, parameter_code))
        if not candidates:
            continue
        values = pd.DataFrame(
            {
                column: pd.to_numeric(raw[column], errors="coerce")
                for column in candidates
            },
            index=raw.index,
        ).reindex(output.index)
        finite = np.isfinite(values.to_numpy(float))
        finite_count = finite.sum(axis=1)
        conflict = finite_count >= 2
        output.loc[:, f"{variable}_series_conflict"] = conflict
        output.loc[:, f"{variable}_conflicting_series_count"] = finite_count * conflict
        output.loc[conflict, f"{variable}_value_status"] = (
            "MULTIPLE_FINITE_SERIES_CONFLICT"
        )
        for row_index in np.flatnonzero(conflict):
            conflicting = [
                candidates[column_index]
                for column_index in np.flatnonzero(finite[row_index])
            ]
            output.iat[
                row_index, output.columns.get_loc(f"{variable}_conflicting_series_ids")
            ] = "|".join(conflicting)
            raw_index = output.index[row_index]
            qualifier_values: dict[str, str | None] = {}
            constituent_provenance: dict[str, dict[str, str | float | None]] = {}
            for column in conflicting:
                qualifier_column = f"{column}_cd"
                if qualifier_column in raw:
                    raw_qualifier = raw.loc[raw_index, qualifier_column]
                    if pd.isna(raw_qualifier) or not str(raw_qualifier).strip():
                        qualifier = None
                    else:
                        qualifier = str(raw_qualifier).strip()
                else:
                    qualifier = None
                raw_value = raw.loc[raw_index, column]
                raw_value_text = None if pd.isna(raw_value) else str(raw_value).strip()
                parsed_value = float(values.loc[raw_index, column])
                if not raw_value_text or not np.isfinite(parsed_value):
                    raise ValueError(
                        "finite NWIS conflict constituent lacks its raw value"
                    )
                qualifier_values[column] = qualifier
                constituent_provenance[column] = {
                    "value_column": column,
                    "qualifier_column": (
                        qualifier_column if qualifier_column in raw else None
                    ),
                    "raw_qualifier": qualifier,
                    "raw_value": raw_value_text,
                    "parsed_finite_value": parsed_value,
                }
            output.iat[
                row_index,
                output.columns.get_loc(
                    f"{variable}_conflicting_series_qualifiers"
                ),
            ] = json.dumps(
                qualifier_values,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            output.iat[
                row_index,
                output.columns.get_loc(
                    f"{variable}_conflicting_series_provenance"
                ),
            ] = json.dumps(
                constituent_provenance,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        for column_index, column in enumerate(candidates):
            selected = (finite_count == 1) & finite[:, column_index]
            if not selected.any():
                continue
            selected_index = output.index[selected]
            output.loc[selected_index, variable] = values.loc[
                selected_index, column
            ].to_numpy(float)
            output.loc[selected_index, f"{variable}_value_status"] = (
                "RETAINED_FINITE_VALUE"
            )
            output.loc[selected_index, f"{variable}_series_id"] = column
            qualifier_column = f"{column}_cd"
            if qualifier_column in raw:
                qualifiers = raw[qualifier_column].astype("string").str.strip()
                qualifiers = qualifiers.mask(qualifiers.eq(""), pd.NA)
                output.loc[selected_index, f"{variable}_qualifier"] = (
                    qualifiers.reindex(selected_index).array
                )
                output.loc[
                    selected_index, f"{variable}_qualifier_column"
                ] = qualifier_column
    output = output.reset_index()
    output.insert(0, "site_no", site_no)
    return output[columns]


def build_nwis_confirmatory_url(site_no: str, start: str, end: str) -> str:
    """Build the one admissible Route-A daily-values request for one site."""
    site_no = str(site_no).strip()
    if not re.fullmatch(r"[0-9]{8,15}", site_no):
        raise ValueError("confirmatory NWIS request has an invalid site_no")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("confirmatory NWIS request interval is reversed")
    query = urlencode({
        "format": "rdb",
        "sites": site_no,
        "startDT": first.strftime("%Y-%m-%d"),
        "endDT": last.strftime("%Y-%m-%d"),
        "parameterCd": "00010,00060,00065",
        "statCd": "00003",
        "siteStatus": "all",
    })
    return f"https://waterservices.usgs.gov/nwis/dv/?{query}"


def fetch_nwis_daily(
    site: str,
    start: str,
    end: str,
    snapshot_store: SnapshotStore | None = None,
) -> pd.DataFrame | None:
    """Daily WTEMP/FLOW/WLEVEL for one site, reindexed to a gap-free calendar."""
    if snapshot_store is None:
        import dataretrieval.nwis as nwis

        try:
            out = nwis.get_record(sites=site, service="dv", start=start, end=end,
                                  parameterCd=list(NWIS_PARAMS))
            raw = out[0] if isinstance(out, tuple) else out
        except Exception:
            return None
    else:
        url = "https://waterservices.usgs.gov/nwis/dv/?" + urlencode({
            "format": "rdb",
            "sites": site,
            "startDT": start,
            "endDT": end,
            "parameterCd": ",".join(NWIS_PARAMS),
            "siteStatus": "all",
        })
        payload, _ = snapshot_store.fetch(provider="usgs-nwis-dv", url=url)
        raw = _parse_nwis_rdb(payload)
        if len(raw) and "datetime" in raw:
            raw = raw.set_index("datetime")
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


def build_daymet_url(lat: float, lon: float, start: str, end: str) -> str:
    """Return the canonical single-pixel request used by the frozen pipeline.

    The Daymet endpoint accepts year-resolution bounds.  The parser below
    subsequently restricts the response to the exact requested calendar.  A
    fixed coordinate representation keeps SnapshotStore request identities
    stable across Python and pandas versions.
    """
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first > last:
        raise ValueError("Daymet request interval is reversed")
    if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Daymet request has invalid coordinates")
    return (
        "https://daymet.ornl.gov/single-pixel/api/data"
        f"?lat={float(lat):.8f}&lon={float(lon):.8f}"
        "&vars=tmax,tmin,prcp,srad,vp"
        f"&start={first.year:04d}-01-01&end={last.year:04d}-12-31"
    )


def parse_daymet_daily(payload: bytes, *, start: str, end: str) -> pd.DataFrame:
    """Parse immutable Daymet bytes into four legacy-schema predictor fields.

    Missing source days/values are retained as NaN on a complete Gregorian
    calendar.  This is deliberate: missingness is an input feature, whereas
    silently dropping dates would change forecast-window identities.
    """
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("Daymet response interval is reversed")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("Daymet response is not UTF-8") from exc
    lines = text.splitlines()
    header = next(
        (index for index, line in enumerate(lines) if line.strip().lower().startswith("year,")),
        None,
    )
    if header is None:
        raise ValueError("Daymet response lacks a CSV header")
    raw = pd.read_csv(io.StringIO("\n".join(lines[header:])))
    required = {
        "year", "yday", "tmax (deg c)", "tmin (deg c)",
        "prcp (mm/day)", "srad (W/m^2)", "vp (Pa)",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Daymet response lacks fields: {sorted(missing)}")
    year = pd.to_numeric(raw["year"], errors="coerce")
    yday = pd.to_numeric(raw["yday"], errors="coerce")
    if (
        year.isna().any()
        or yday.isna().any()
        # Daymet's documented calendar is always exactly yday 1..365.  Leap
        # years retain February 29 and omit December 31.
        or not yday.between(1, 365).all()
        or not np.equal(year, np.floor(year)).all()
        or not np.equal(yday, np.floor(yday)).all()
    ):
        raise ValueError("Daymet response contains an invalid year/day key")
    date = (
        pd.to_datetime(year.astype(int).astype(str) + "-01-01", errors="coerce")
        + pd.to_timedelta(yday.astype(int) - 1, unit="D")
    )
    if date.isna().any() or date.duplicated().any():
        raise ValueError("Daymet response contains invalid or duplicate daily keys")

    def numeric(column: str) -> np.ndarray:
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
        values[(~np.isfinite(values)) | (values <= -9990.0)] = np.nan
        return values

    tmax = numeric("tmax (deg c)")
    tmin = numeric("tmin (deg c)")
    tmean = (tmax + tmin) / 2.0
    tmean[tmax < tmin] = np.nan
    prcp = numeric("prcp (mm/day)")
    radiation = numeric("srad (W/m^2)")
    vapour_pressure = numeric("vp (Pa)")
    humidity = 100.0 * vapour_pressure / _svp(tmean)
    # Negative precipitation/radiation are provider fill or invalid values, not
    # physically meaningful forcings.  Preserve them as explicit missingness.
    prcp[prcp < 0] = np.nan
    radiation[radiation < 0] = np.nan
    meteorology = pd.DataFrame(
        {
            "TEMP": tmean,
            "PRCP": prcp,
            "RHMEAN": np.clip(humidity, 0.0, 100.0),
            "DH": radiation,
        },
        index=pd.DatetimeIndex(date, name="DATE"),
    )
    full = pd.date_range(first, last, freq="D", name="DATE")
    return meteorology.reindex(full)[["TEMP", "PRCP", "RHMEAN", "DH"]]


def fetch_daymet(lat: float, lon: float, start: str, end: str,
                 retries: int = 3,
                 snapshot_store: SnapshotStore | None = None) -> pd.DataFrame | None:
    """Daily TEMP/PRCP/DH/RHMEAN legacy-schema proxies at a Daymet pixel.

    Daymet covers 1980–present.  Its 365-day leap-year convention can omit one
    Gregorian date, so we reindex to the exact full calendar and retain that
    absence for the downstream missingness-aware preprocessing.
    """
    url = build_daymet_url(lat, lon, start, end)
    raw = None
    if snapshot_store is not None:
        payload, _ = snapshot_store.fetch(
            provider="ornl-daymet-single-pixel", url=url, retries=retries)
        raw = payload.decode("utf-8")
    else:
        for _ in range(retries):
            try:
                raw = urllib.request.urlopen(url, timeout=60).read().decode()
                break
            except Exception:
                time.sleep(1.0)
    if raw is None:
        return None
    try:
        return parse_daymet_daily(raw.encode("utf-8"), start=start, end=end)
    except (ValueError, pd.errors.ParserError):
        return None


# --------------------------------------------------------------------------- #
# gridMET wind speed (point, via NWK THREDDS NetCDF Subset Service)
# --------------------------------------------------------------------------- #
def build_gridmet_wind_url(lat: float, lon: float, start: str, end: str) -> str:
    """Return the canonical gridMET point-subset request URL."""
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("gridMET request interval is reversed")
    if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("gridMET request has invalid coordinates")
    return (
        "https://thredds.northwestknowledge.net/thredds/ncss/"
        f"{GRIDMET_WIND_DATASET}?var=daily_mean_wind_speed"
        f"&latitude={float(lat):.8f}&longitude={float(lon):.8f}"
        f"&time_start={first:%Y-%m-%d}T00:00:00Z"
        f"&time_end={last:%Y-%m-%d}T00:00:00Z&accept=csv"
    )


def build_gridmet_wind_metadata_url() -> str:
    """Return the authoritative OPeNDAP attribute document for the wind field."""
    return (
        "https://thredds.northwestknowledge.net/thredds/dodsC/"
        f"{GRIDMET_WIND_DATASET}.das"
    )


def parse_gridmet_wind_metadata(payload: bytes) -> dict[str, float | str]:
    """Validate the provider-declared packing contract used by NCSS CSV.

    NCSS returns the stored integer values while labelling the column ``m/s``.
    The dataset's OPeNDAP DAS is therefore required evidence for applying the
    CF ``scale_factor`` and ``add_offset``.  Failing to find one unambiguous
    variable block is a hard error rather than an invitation to guess units.
    """
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("gridMET metadata is not UTF-8") from exc
    match = re.search(
        r"daily_mean_wind_speed\s*\{(?P<body>.*?)\n\s*\}",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("gridMET metadata lacks daily_mean_wind_speed")
    body = match.group("body")

    def one(pattern: str, label: str) -> str:
        values = re.findall(pattern, body)
        if len(values) != 1:
            raise ValueError(f"gridMET metadata has ambiguous {label}")
        return str(values[0])

    units = one(r'String\s+units\s+"([^"]+)"\s*;', "units")
    scale = float(one(r"Float(?:32|64)\s+scale_factor\s+([-+0-9.eE]+)\s*;", "scale_factor"))
    offset = float(one(r"Float(?:32|64)\s+add_offset\s+([-+0-9.eE]+)\s*;", "add_offset"))
    if units != "m/s":
        raise ValueError(f"gridMET wind units changed: {units!r}")
    if not np.isclose(scale, GRIDMET_WIND_SCALE_FACTOR, rtol=0.0, atol=0.0):
        raise ValueError(f"gridMET wind scale_factor changed: {scale}")
    if not np.isclose(offset, GRIDMET_WIND_ADD_OFFSET, rtol=0.0, atol=0.0):
        raise ValueError(f"gridMET wind add_offset changed: {offset}")
    return {"units": units, "scale_factor": scale, "add_offset": offset}


def parse_gridmet_wind_daily(
    payload: bytes,
    *,
    start: str,
    end: str,
    scale_factor: float = GRIDMET_WIND_SCALE_FACTOR,
    add_offset: float = GRIDMET_WIND_ADD_OFFSET,
) -> pd.Series:
    """Apply a previously verified CF packing contract to gridMET NCSS CSV."""
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if first > last:
        raise ValueError("gridMET response interval is reversed")
    try:
        raw = pd.read_csv(io.BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("gridMET response is not valid CSV") from exc
    time_columns = [column for column in raw if str(column).strip().lower() == "time"]
    wind_columns = [column for column in raw if "wind" in str(column).lower()]
    if len(time_columns) != 1 or len(wind_columns) != 1:
        raise ValueError("gridMET response lacks an unambiguous time/wind schema")
    dates = pd.to_datetime(raw[time_columns[0]], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("gridMET response contains an invalid timestamp")
    dates = dates.dt.tz_convert(None).dt.normalize()
    if dates.duplicated().any():
        raise ValueError("gridMET response duplicates a daily key")
    if not np.isclose(scale_factor, GRIDMET_WIND_SCALE_FACTOR, rtol=0.0, atol=0.0):
        raise ValueError("unverified gridMET wind scale_factor")
    if not np.isclose(add_offset, GRIDMET_WIND_ADD_OFFSET, rtol=0.0, atol=0.0):
        raise ValueError("unverified gridMET wind add_offset")
    packed = pd.to_numeric(raw[wind_columns[0]], errors="coerce").to_numpy(float)
    values = packed * float(scale_factor) + float(add_offset)
    values[(values < 0.0) | (values > 100.0)] = np.nan
    wind = pd.Series(values, index=pd.DatetimeIndex(dates), name="WDSP")
    full = pd.date_range(first, last, freq="D", name="DATE")
    return wind.reindex(full)


def fetch_gridmet_wind(lat: float, lon: float, start: str, end: str,
                       retries: int = 3,
                       snapshot_store: SnapshotStore | None = None) -> pd.Series | None:
    """Daily mean wind speed at a point from gridMET (CONUS, 1979–present).

    NCSS returns the packed integer field.  The frozen acquisition separately
    archives and validates the dataset DAS declaring ``scale_factor=0.1``,
    ``add_offset=0`` and ``units=m/s`` before these values are consumed.
    """
    url = build_gridmet_wind_url(lat, lon, start, end)
    raw = None
    if snapshot_store is not None:
        payload, _ = snapshot_store.fetch(
            provider="gridmet-ncss", url=url, retries=retries)
        raw = payload.decode("utf-8")
    else:
        for _ in range(retries):
            try:
                raw = urllib.request.urlopen(url, timeout=60).read().decode()
                break
            except Exception:
                time.sleep(1.0)
    if raw is None:
        return None
    try:
        return parse_gridmet_wind_daily(raw.encode("utf-8"), start=start, end=end)
    except (ValueError, pd.errors.ParserError):
        return None


# --------------------------------------------------------------------------- #
# Assemble one station into the study schema
# --------------------------------------------------------------------------- #
def build_station(site: str, lat: float, lon: float, site_id: str,
                  start: str, end: str, min_wtemp_cov: float = 0.0,
                  min_flow_cov: float = 0.0,
                  min_wtemp_cov_test: float = 0.0,
                  min_flow_cov_test: float = 0.0,
                  test_start: str = "2019-01-01",
                  snapshot_store: SnapshotStore | None = None,
                  ) -> tuple[pd.DataFrame | None, dict]:
    """Acquire one station and apply the inclusion thresholds.

    Two coverage gates are checked, both BEFORE the (slower) Daymet/gridMET
    calls so probing stays fast:

    * **Full-period** (``min_wtemp_cov`` / ``min_flow_cov``): observation
      density across the whole 2006–2020 record.
    * **Development-evaluation period** (legacy argument names
      ``min_wtemp_cov_test`` / ``min_flow_cov_test``): observation density from
      2019 onward.  Because this gate looked at 2019--2020 availability, those
      years are exploratory development evidence, not a blind/untouched test.
    """
    nwis_df = fetch_nwis_daily(site, start, end, snapshot_store=snapshot_store)
    if nwis_df is None:
        return None, {"site": site, "ok": False, "reason": "no NWIS WTEMP+FLOW"}
    wt_cov = nwis_df["WTEMP"].notna().mean()
    fl_cov = nwis_df["FLOW"].notna().mean()
    test_mask = nwis_df.index >= pd.Timestamp(test_start)
    wt_cov_test = nwis_df.loc[test_mask, "WTEMP"].notna().mean() if test_mask.any() else 0.0
    fl_cov_test = nwis_df.loc[test_mask, "FLOW"].notna().mean() if test_mask.any() else 0.0
    cov_info = {"wtemp_cov": round(float(wt_cov), 3),
                "flow_cov": round(float(fl_cov), 3),
                "wtemp_cov_test": round(float(wt_cov_test), 3),
                "flow_cov_test": round(float(fl_cov_test), 3)}
    # reject on coverage BEFORE the (slower) Daymet call to speed up probing
    if wt_cov < min_wtemp_cov or fl_cov < min_flow_cov:
        return None, {"site": site, "ok": False, "reason": "low full-period coverage",
                      **cov_info}
    if (wt_cov_test < min_wtemp_cov_test) or (fl_cov_test < min_flow_cov_test):
        return None, {"site": site, "ok": False,
                      "reason": "low development-evaluation-period coverage", **cov_info}
    met = fetch_daymet(lat, lon, start, end, snapshot_store=snapshot_store)
    if met is None:
        return None, {"site": site, "ok": False, "reason": "no Daymet"}
    df = nwis_df.join(met, how="left")
    if "WLEVEL" not in df:
        df["WLEVEL"] = np.nan
    wind = fetch_gridmet_wind(
        lat, lon, start, end, snapshot_store=snapshot_store)  # gridMET wind index
    df["WDSP"] = wind.reindex(df.index).to_numpy() if wind is not None else np.nan
    df = df.reset_index().rename(columns={"index": "DATE"})
    df.insert(1, "site_id", site_id)
    df = df[["DATE", "site_id", "WTEMP", "FLOW", "WLEVEL",
             "TEMP", "PRCP", "WDSP", "RHMEAN", "DH"]]
    info = {"site": site, "site_id": site_id, "ok": True,
            "wtemp_cov": round(float(wt_cov), 3), "flow_cov": round(float(fl_cov), 3),
            "wtemp_cov_test": round(float(wt_cov_test), 3),
            "flow_cov_test": round(float(fl_cov_test), 3),
            "wlevel_cov": round(float(df["WLEVEL"].notna().mean()), 3),
            "n_days": len(df)}
    return df, info
