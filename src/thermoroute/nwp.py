"""Optional archived-GFS sensitivity inputs for Route A.

Open-Meteo's Previous Runs API aligns archived model values by a fixed offset
from each *valid hour*.  It does not identify a single initialization run for a
whole target day.  We therefore aggregate each UTC target day as a rolling
fixed-lead composite and state that limitation in every derived row.

The current Route-A primary models do not consume these horizon-specific values.
They are a secondary availability/sensitivity artifact only.  No function in
this module accepts or emits water-temperature outcomes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
import json
from urllib.parse import urlencode

import numpy as np
import pandas as pd


PREVIOUS_RUNS_ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL = "gfs_global"
ROUTE_A_LEADS = (1, 3, 7)
GFS_ARCHIVE_RUN_START = date(2021, 3, 23)
NWP_COMMON_VALID_TIME_START = date(2021, 3, 30)
ISSUE_SEMANTICS = (
    "rolling_fixed_valid_time_minus_lead_offset; daily composite available by "
    "23:59 UTC on issue_date; not a single model-run initialization"
)


class NWPContractError(RuntimeError):
    """Raised when an archived predictor response violates the frozen contract."""


def _iso_date(value: str | date | pd.Timestamp) -> date:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc
    if parsed.tzinfo is not None or parsed != parsed.normalize():
        raise ValueError(f"expected a timezone-free calendar date: {value!r}")
    return parsed.date()


def _coordinate(value: float, *, latitude: bool) -> str:
    number = float(value)
    lower, upper = (-90.0, 90.0) if latitude else (-180.0, 180.0)
    if not np.isfinite(number) or not lower <= number <= upper:
        axis = "latitude" if latitude else "longitude"
        raise ValueError(f"invalid {axis}: {value!r}")
    # Six decimal places are sub-metre to decimetre scale and make request
    # fingerprints independent of platform-specific float repr details.
    return f"{number:.6f}"


def build_previous_runs_url(
    *,
    latitude: float,
    longitude: float,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
    leads: Sequence[int] = ROUTE_A_LEADS,
) -> str:
    """Build a canonical, one-location GFS fixed-lead request URL."""
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    fixed_leads = tuple(sorted({int(lead) for lead in leads}))
    if fixed_leads != ROUTE_A_LEADS:
        raise ValueError(f"Route A requires exactly leads {ROUTE_A_LEADS}")
    hourly = ",".join(
        f"temperature_2m_previous_day{lead}" for lead in fixed_leads
    )
    params = {
        "end_date": end.isoformat(),
        "hourly": hourly,
        "latitude": _coordinate(latitude, latitude=True),
        "longitude": _coordinate(longitude, latitude=False),
        "models": OPEN_METEO_MODEL,
        "start_date": start.isoformat(),
        "temperature_unit": "celsius",
        "timezone": "GMT",
    }
    return PREVIOUS_RUNS_ENDPOINT + "?" + urlencode(sorted(params.items()))


def iter_month_chunks(
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp,
) -> Iterator[tuple[date, date]]:
    """Yield deterministic calendar-month request blocks, inclusive at both ends."""
    start = _iso_date(start_date)
    end = _iso_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    cursor = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cursor <= final:
        month_end = cursor + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, final)
        yield cursor.date(), chunk_end.date()
        cursor = chunk_end + pd.Timedelta(days=1)


def _load_response(payload: bytes) -> dict[str, object]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NWPContractError("Previous Runs response is not valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise NWPContractError("Previous Runs response must be one-location JSON object")
    return document


def parse_previous_runs_daily(
    payload: bytes,
    *,
    site_no: str,
    requested_start: str | date | pd.Timestamp,
    requested_end: str | date | pd.Timestamp,
    leads: Sequence[int] = ROUTE_A_LEADS,
) -> pd.DataFrame:
    """Validate and aggregate one raw response to complete UTC target days.

    A day with fewer than 24 finite hourly values is retained as unavailable;
    its daily mean is ``NaN``.  This prevents a partial-day average from silently
    becoming a predictor and preserves pre-label availability diagnostics.
    """
    fixed_leads = tuple(sorted({int(lead) for lead in leads}))
    if fixed_leads != ROUTE_A_LEADS:
        raise ValueError(f"Route A requires exactly leads {ROUTE_A_LEADS}")
    site = str(site_no).strip()
    if not site:
        raise ValueError("site_no must be non-empty")
    start = _iso_date(requested_start)
    end = _iso_date(requested_end)
    if start > end:
        raise ValueError("requested_start must be on or before requested_end")

    document = _load_response(payload)
    if document.get("timezone") not in {"GMT", "UTC"}:
        raise NWPContractError("Previous Runs response is not in frozen GMT/UTC timezone")
    utc_offset = document.get("utc_offset_seconds", 0)
    if not isinstance(utc_offset, (str, int, float)) or int(utc_offset) != 0:
        raise NWPContractError("Previous Runs response has a non-zero UTC offset")
    hourly = document.get("hourly")
    units = document.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise NWPContractError("Previous Runs response lacks hourly data or units")
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list):
        raise NWPContractError("Previous Runs response lacks hourly time array")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(raw_times, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise NWPContractError("Previous Runs response contains an invalid time") from exc
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    expected = pd.date_range(
        pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h"
    )
    if not times.equals(expected):
        raise NWPContractError(
            "Previous Runs hourly times do not exactly cover the requested UTC days"
        )

    response_lat = pd.to_numeric(document.get("latitude"), errors="coerce")
    response_lon = pd.to_numeric(document.get("longitude"), errors="coerce")
    if not np.isfinite(response_lat) or not np.isfinite(response_lon):
        raise NWPContractError("Previous Runs response lacks finite grid coordinates")

    records: list[dict[str, object]] = []
    target_dates = pd.DatetimeIndex(times.normalize().unique())
    for lead in fixed_leads:
        field = f"temperature_2m_previous_day{lead}"
        values = hourly.get(field)
        if not isinstance(values, list) or len(values) != len(times):
            raise NWPContractError(f"Previous Runs response lacks complete array {field}")
        if units.get(field) != "°C":
            raise NWPContractError(f"Previous Runs response has unexpected unit for {field}")
        invalid_values = [
            value for value in values
            if value is not None
            and (isinstance(value, bool) or not isinstance(value, (int, float)))
        ]
        if invalid_values:
            raise NWPContractError(f"Previous Runs response has non-numeric values in {field}")
        numeric = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce")
        series = pd.Series(numeric.to_numpy(dtype=float), index=times)
        for target in target_dates:
            day_values = series.loc[target:target + pd.Timedelta(hours=23)]
            finite = np.isfinite(day_values.to_numpy())
            available = int(finite.sum())
            complete = available == 24
            records.append({
                "site_no": site,
                "horizon": lead,
                "issue_date": target - pd.Timedelta(days=lead),
                "target_date": target,
                "air_temp_2m_mean_c": (
                    float(day_values.mean()) if complete else float("nan")
                ),
                "available_hour_count": available,
                "complete_target_day": complete,
                "lead_field": field,
                "source_provider": "Open-Meteo Previous Runs API",
                "upstream_model": "NOAA NCEP GFS global",
                "open_meteo_model": OPEN_METEO_MODEL,
                "issue_semantics": ISSUE_SEMANTICS,
                "response_grid_lat": float(response_lat),
                "response_grid_lon": float(response_lon),
            })
    out = pd.DataFrame.from_records(records)
    out = out.sort_values(["site_no", "horizon", "target_date"]).reset_index(drop=True)
    key = ["site_no", "horizon", "issue_date", "target_date"]
    if out.duplicated(key).any():
        raise NWPContractError("derived predictor rows duplicate the forecast key")
    return out
