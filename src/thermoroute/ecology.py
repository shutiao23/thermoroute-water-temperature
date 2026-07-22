"""Auditable daily-maximum and 7DADM utilities.

The model target in Route A is daily *mean* water temperature.  These functions
exist to prevent that target from being mislabeled as a regulatory 7DADM outcome.
They require an explicitly named daily-maximum column and seven consecutive days.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_COLUMNS = {
    "site_no",
    "jurisdiction",
    "designated_use",
    "species_life_stage",
    "season_start",
    "season_end",
    "threshold_c",
    "source_url",
}


def compute_7dadm(daily_maximum: pd.DataFrame, *, date_col: str = "DATE",
                   station_col: str = "site_no", maximum_col: str = "WTEMP_MAX"
                   ) -> pd.DataFrame:
    """Compute the 7-day average of daily maxima on consecutive daily records."""
    required = {date_col, station_col, maximum_col}
    missing = required - set(daily_maximum)
    if missing:
        raise ValueError(f"daily-maximum data missing columns: {sorted(missing)}")
    if "mean" in maximum_col.lower() or maximum_col == "WTEMP":
        raise ValueError("7DADM requires an explicitly identified daily-maximum field")
    pieces = []
    for station, group in daily_maximum.groupby(station_col):
        group = group[[date_col, maximum_col]].copy()
        group[date_col] = pd.to_datetime(group[date_col]).dt.normalize()
        if group[date_col].duplicated().any():
            raise ValueError(f"duplicate daily maximum for station {station}")
        group = group.set_index(date_col).sort_index()
        calendar = pd.date_range(group.index.min(), group.index.max(), freq="D")
        values = pd.to_numeric(group[maximum_col], errors="coerce").reindex(calendar)
        seven = values.rolling(window=7, min_periods=7).mean()
        pieces.append(pd.DataFrame({
            station_col: str(station),
            date_col: calendar,
            maximum_col: values.to_numpy(),
            "SEVEN_DADM": seven.to_numpy(),
            "SEVEN_DADM_N": values.notna().rolling(window=7, min_periods=1).sum().to_numpy(int),
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=[station_col, date_col, maximum_col, "SEVEN_DADM", "SEVEN_DADM_N"]
    )


def validate_standard_registry(standards: pd.DataFrame) -> None:
    missing = STANDARD_COLUMNS - set(standards)
    if missing:
        raise ValueError(f"ecological standard registry missing: {sorted(missing)}")
    if standards.empty:
        raise ValueError("ecological standard registry is empty")
    if standards[list(STANDARD_COLUMNS)].isna().any().any():
        raise ValueError("ecological standard registry contains missing required values")
    threshold = pd.to_numeric(standards.threshold_c, errors="coerce")
    if threshold.isna().any() or not threshold.between(0, 40).all():
        raise ValueError("threshold_c must be a plausible Celsius value")
    for column in ("season_start", "season_end"):
        try:
            pd.to_datetime("2001-" + standards[column].astype(str), format="%Y-%m-%d")
        except ValueError as error:
            raise ValueError(f"{column} must use MM-DD") from error
    if not standards.source_url.astype(str).str.startswith(("https://", "http://")).all():
        raise ValueError("every standard requires a public source URL")


def load_standard_registry(path: str | Path) -> pd.DataFrame:
    standards = pd.read_csv(path, dtype={"site_no": str})
    standards["site_no"] = standards.site_no.str.zfill(8)
    validate_standard_registry(standards)
    return standards


def active_standard_mask(dates: Iterable, start_mmdd: str, end_mmdd: str) -> np.ndarray:
    """Return seasonal applicability, supporting seasons that cross New Year."""
    mmdd = pd.to_datetime(dates).strftime("%m-%d")
    if start_mmdd <= end_mmdd:
        return np.asarray((mmdd >= start_mmdd) & (mmdd <= end_mmdd))
    return np.asarray((mmdd >= start_mmdd) | (mmdd <= end_mmdd))
