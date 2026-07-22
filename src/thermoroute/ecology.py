"""Auditable *observed* daily-maximum and 7DADM utilities.

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

STANDARD_CONTEXT_COLUMNS = [
    "site_no",
    "jurisdiction",
    "designated_use",
    "species_life_stage",
    "season_start",
    "season_end",
]

STANDARD_SUBJECT_COLUMNS = [
    "site_no",
    "jurisdiction",
    "designated_use",
    "species_life_stage",
]

OBSERVED_THRESHOLD_COLUMNS = [
    "site_no",
    "DATE",
    "WTEMP_MAX",
    "SEVEN_DADM",
    "SEVEN_DADM_N",
    "jurisdiction",
    "designated_use",
    "species_life_stage",
    "season_start",
    "season_end",
    "threshold_c",
    "source_url",
    "applicable",
    "comparison_status",
    "exceedance",
]


def _normalise_site_numbers(values: Iterable, *, source: str) -> pd.Series:
    sites = pd.Series(values, dtype="string").str.strip()
    if sites.isna().any() or sites.eq("").any():
        raise ValueError(f"{source} contains a missing site_no")
    if not sites.str.fullmatch(r"\d+").all():
        raise ValueError(f"{source} site_no values must contain digits only")
    return sites.str.zfill(8)


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
    daily_maximum = daily_maximum.copy()
    daily_maximum[station_col] = _normalise_site_numbers(
        daily_maximum[station_col], source="daily-maximum data"
    )
    pieces = []
    for station, group in daily_maximum.groupby(station_col):
        group = group[[date_col, maximum_col]].copy()
        group[date_col] = pd.to_datetime(group[date_col], errors="raise").dt.normalize()
        if group[date_col].duplicated().any():
            raise ValueError(f"duplicate daily maximum for station {station}")
        group = group.set_index(date_col).sort_index()
        numeric = pd.to_numeric(group[maximum_col], errors="coerce")
        if numeric.isna().sum() > group[maximum_col].isna().sum():
            raise ValueError(f"non-numeric daily maximum for station {station}")
        if np.isinf(numeric).any():
            raise ValueError(f"non-finite daily maximum for station {station}")
        calendar = pd.date_range(group.index.min(), group.index.max(), freq="D")
        values = numeric.reindex(calendar)
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
    normalised = standards.copy()
    normalised["site_no"] = _normalise_site_numbers(
        normalised["site_no"], source="ecological standard registry"
    )
    text_columns = [
        "jurisdiction", "designated_use", "species_life_stage", "source_url"
    ]
    if normalised[text_columns].astype("string").apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError("ecological standard registry contains an empty context value")
    threshold = pd.to_numeric(standards.threshold_c, errors="coerce")
    if threshold.isna().any() or not threshold.between(0, 40).all():
        raise ValueError("threshold_c must be a plausible Celsius value")
    for column in ("season_start", "season_end"):
        values = normalised[column].astype("string")
        if not values.str.fullmatch(r"(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])").all():
            raise ValueError(f"{column} must use zero-padded MM-DD")
        try:
            pd.to_datetime("2000-" + values, format="%Y-%m-%d")
        except ValueError as error:
            raise ValueError(f"{column} must use MM-DD") from error
    if not normalised.source_url.astype(str).str.startswith(("https://", "http://")).all():
        raise ValueError("every standard requires a public source URL")
    if normalised.duplicated(STANDARD_CONTEXT_COLUMNS, keep=False).any():
        raise ValueError(
            "ecological standard registry has ambiguous duplicate context/season records"
        )
    calendar = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    for context, group in normalised.groupby(STANDARD_SUBJECT_COLUMNS, dropna=False):
        occupied = np.zeros(len(calendar), dtype=bool)
        for row in group.itertuples(index=False):
            active = active_standard_mask(calendar, row.season_start, row.season_end)
            if np.any(occupied & active):
                raise ValueError(
                    "ecological standard registry has overlapping seasons for "
                    f"context={context}"
                )
            occupied |= active


def load_standard_registry(path: str | Path) -> pd.DataFrame:
    standards = pd.read_csv(path, dtype={"site_no": str})
    validate_standard_registry(standards)
    standards["site_no"] = _normalise_site_numbers(
        standards["site_no"], source="ecological standard registry"
    )
    standards["threshold_c"] = pd.to_numeric(standards["threshold_c"])
    return standards


def active_standard_mask(dates: Iterable, start_mmdd: str, end_mmdd: str) -> np.ndarray:
    """Return seasonal applicability, supporting seasons that cross New Year."""
    mmdd = pd.to_datetime(dates).strftime("%m-%d")
    if start_mmdd <= end_mmdd:
        return np.asarray((mmdd >= start_mmdd) & (mmdd <= end_mmdd))
    return np.asarray((mmdd >= start_mmdd) | (mmdd <= end_mmdd))


def _validate_observed_7dadm(seven: pd.DataFrame) -> pd.DataFrame:
    """Verify that a table is the strict rolling result of its observed maxima."""
    required = {"site_no", "DATE", "WTEMP_MAX", "SEVEN_DADM", "SEVEN_DADM_N"}
    missing = required - set(seven)
    if missing:
        raise ValueError(f"observed 7DADM table missing columns: {sorted(missing)}")
    observed = seven[
        ["site_no", "DATE", "WTEMP_MAX", "SEVEN_DADM", "SEVEN_DADM_N"]
    ].copy()
    observed["site_no"] = _normalise_site_numbers(
        observed["site_no"], source="observed 7DADM table"
    )
    observed["DATE"] = pd.to_datetime(observed["DATE"], errors="raise").dt.normalize()
    if observed.duplicated(["site_no", "DATE"]).any():
        raise ValueError("observed 7DADM table has duplicate site/date rows")

    expected = compute_7dadm(observed[["site_no", "DATE", "WTEMP_MAX"]])
    expected["site_no"] = _normalise_site_numbers(
        expected["site_no"], source="computed observed 7DADM table"
    )
    order = ["site_no", "DATE"]
    observed = observed.sort_values(order).reset_index(drop=True)
    expected = expected.sort_values(order).reset_index(drop=True)
    if not observed[order].equals(expected[order]):
        raise ValueError("observed 7DADM table does not cover each consecutive calendar day")
    supplied_n = pd.to_numeric(observed["SEVEN_DADM_N"], errors="coerce")
    if supplied_n.isna().any() or not np.array_equal(
        supplied_n.to_numpy(float), expected["SEVEN_DADM_N"].to_numpy(float)
    ):
        raise ValueError("observed 7DADM counts do not match the daily maxima")
    supplied_series = pd.to_numeric(observed["SEVEN_DADM"], errors="coerce")
    if supplied_series.isna().sum() > observed["SEVEN_DADM"].isna().sum():
        raise ValueError("observed 7DADM table contains a non-numeric value")
    supplied = supplied_series.to_numpy(float)
    calculated = expected["SEVEN_DADM"].to_numpy(float)
    if not np.allclose(supplied, calculated, rtol=0.0, atol=1e-12, equal_nan=True):
        raise ValueError("observed 7DADM values do not match the daily maxima")
    return expected


def apply_observed_standards(
    seven: pd.DataFrame, standards: pd.DataFrame
) -> pd.DataFrame:
    """Apply contextual standards to observed 7DADM values only.

    One output row is emitted per observed site/date and registry context. Seasonal
    applicability is determined from the *ending date* of the seven-day window.
    A missing 7DADM or a date outside a context's declared season is deliberately
    left without an exceedance classification. Site sets must match exactly so a
    missing or unrelated registry cannot silently produce reassuring results.
    """
    observed = _validate_observed_7dadm(seven)
    validate_standard_registry(standards)
    registry = standards.copy()
    registry["site_no"] = _normalise_site_numbers(
        registry["site_no"], source="ecological standard registry"
    )
    registry["threshold_c"] = pd.to_numeric(registry["threshold_c"])

    observed_sites = set(observed["site_no"])
    registry_sites = set(registry["site_no"])
    missing = sorted(observed_sites - registry_sites)
    unrelated = sorted(registry_sites - observed_sites)
    if missing or unrelated:
        raise ValueError(
            "observed/standard site sets do not match exactly; "
            f"missing standards={missing}, registry-only sites={unrelated}"
        )

    applied = observed.merge(registry, on="site_no", how="inner", validate="many_to_many")
    mmdd = applied["DATE"].dt.strftime("%m-%d")
    regular = applied["season_start"] <= applied["season_end"]
    applied["applicable"] = np.where(
        regular,
        (mmdd >= applied["season_start"]) & (mmdd <= applied["season_end"]),
        (mmdd >= applied["season_start"]) | (mmdd <= applied["season_end"]),
    )
    valid = applied["SEVEN_DADM"].notna()
    comparable = valid & applied["applicable"]
    above = applied["SEVEN_DADM"] > applied["threshold_c"]
    applied["comparison_status"] = "insufficient_observed_daily_maxima"
    applied.loc[valid & ~applied["applicable"], "comparison_status"] = (
        "outside_declared_season"
    )
    applied.loc[comparable & ~above, "comparison_status"] = (
        "observed_at_or_below_threshold"
    )
    applied.loc[comparable & above, "comparison_status"] = "observed_above_threshold"
    applied["exceedance"] = pd.Series(pd.NA, index=applied.index, dtype="boolean")
    applied.loc[comparable, "exceedance"] = above.loc[comparable].to_numpy()
    return applied[OBSERVED_THRESHOLD_COLUMNS].sort_values(
        STANDARD_CONTEXT_COLUMNS + ["DATE"]
    ).reset_index(drop=True)


def evaluate_observed_7dadm(
    daily_maximum: pd.DataFrame, standards: pd.DataFrame
) -> pd.DataFrame:
    """Compute observed 7DADM and apply a complete, matching standards registry."""
    return apply_observed_standards(compute_7dadm(daily_maximum), standards)
