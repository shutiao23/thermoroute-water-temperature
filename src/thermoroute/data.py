"""Leakage-safe data loading, quality control, splitting and imputation.

Design rules enforced here (the things reviewers actually check):

* Sentinel codes (WDSP 999.9, PRCP 99.99) are masked to NaN, never used as
  extremes.
* The time split is by calendar date; the blind-test years are isolated.
* Imputation statistics and any per-station scaling are fit on the **training
  fold only** and then applied forward — no future information leaks back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Loading + quality control
# --------------------------------------------------------------------------- #
def _load_one(station: str) -> pd.DataFrame:
    path = C.DATA_RAW / C.RAW_FILES[station]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # The raw CSVs carry stray whitespace inside numeric cells (e.g. "5.2 ").
    for col in df.columns:
        if col == "DATE":
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df.insert(1, "site_id", station)
    return df


def _mask_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace sentinel missing-codes with NaN; return how many were masked."""
    counts: dict[str, int] = {}
    out = df.copy()
    for var, code in C.SENTINELS.items():
        if var in out.columns:
            hit = out[var] >= code
            counts[var] = int(hit.sum())
            out.loc[hit, var] = np.nan
    return out, counts


def load_panel() -> pd.DataFrame:
    """Return the long panel ``[DATE, site_id, <vars...>]`` with sentinels masked.

    A boolean ``<var>_observed`` column records, per cell, whether the value was
    genuinely observed (used by mask-aware models and for honest QC reporting).
    """
    frames = []
    for st in C.STATIONS:
        df, _ = _mask_sentinels(_load_one(st))
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["site_id", "DATE"]).reset_index(drop=True)
    for var in C.ALL_VARS:
        panel[f"{var}_observed"] = panel[var].notna()
    return panel


def sentinel_report() -> pd.DataFrame:
    """Per-station count of masked sentinel cells (for the QC table)."""
    rows = []
    for st in C.STATIONS:
        _, counts = _mask_sentinels(_load_one(st))
        rows.append({"site_id": st, **counts})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Time split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitMasks:
    """Boolean masks over a DATE index for each split partition."""
    train: np.ndarray
    val: np.ndarray
    calib: np.ndarray
    test: np.ndarray


def split_masks(dates: pd.Series, split: C.TimeSplit = C.SPLIT) -> SplitMasks:
    d = pd.to_datetime(dates).to_numpy()

    def m(lo: str, hi: str) -> np.ndarray:
        return (d >= np.datetime64(lo)) & (d <= np.datetime64(hi))

    s = split.as_dict()
    return SplitMasks(
        train=m(*s["train"]), val=m(*s["val"]),
        calib=m(*s["calib"]), test=m(*s["test"]),
    )


def assert_split_disjoint(split: C.TimeSplit = C.SPLIT) -> None:
    """Fail loudly if partitions overlap or are out of chronological order."""
    bounds = [split.train, split.val, split.calib, split.test]
    for (a_lo, a_hi), (b_lo, b_hi) in zip(bounds, bounds[1:]):
        assert a_hi < b_lo, f"split overlap / disorder: {a_hi} !< {b_lo}"


# --------------------------------------------------------------------------- #
# Fold-safe imputation
# --------------------------------------------------------------------------- #
@dataclass
class Imputer:
    """Day-of-year seasonal-median imputation fit on the training fold only.

    The series here are gap-free apart from the masked sentinels, so this only
    fills the handful of WDSP / PRCP holes — but it does so with a statistic that
    never sees the validation/test years, which is the property we need.
    """
    medians: dict[tuple[str, str], pd.Series]  # (site, var) -> Series indexed by doy
    global_median: dict[tuple[str, str], float]

    @classmethod
    def fit(cls, panel: pd.DataFrame, train_mask: np.ndarray) -> "Imputer":
        tr = panel.loc[train_mask].copy()
        tr["doy"] = pd.to_datetime(tr["DATE"]).dt.dayofyear
        medians: dict[tuple[str, str], pd.Series] = {}
        gmed: dict[tuple[str, str], float] = {}
        for st in C.STATIONS:
            sub = tr[tr.site_id == st]
            for var in C.ALL_VARS:
                medians[(st, var)] = sub.groupby("doy")[var].median()
                gmed[(st, var)] = float(sub[var].median())
        return cls(medians=medians, global_median=gmed)

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        doy = pd.to_datetime(out["DATE"]).dt.dayofyear.to_numpy()
        for st in C.STATIONS:
            sel = (out.site_id == st).to_numpy()
            for var in C.ALL_VARS:
                # copy=True: newer pandas (copy-on-write) can return a
                # read-only view here, and col[miss] = fill writes into it.
                col = out[var].to_numpy(dtype=float, copy=True)
                miss = sel & np.isnan(col)
                if not miss.any():
                    continue
                med = self.medians[(st, var)]
                fill = pd.Series(doy[miss]).map(med).to_numpy()
                fill = np.where(np.isnan(fill), self.global_median[(st, var)], fill)
                col[miss] = fill
                out[var] = col
        return out


# --------------------------------------------------------------------------- #
# Per-station standardisation (train-fit) for the deep models
# --------------------------------------------------------------------------- #
@dataclass
class StandardScalerPerStation:
    mean: dict[tuple[str, str], float]
    std: dict[tuple[str, str], float]

    @classmethod
    def fit(cls, panel: pd.DataFrame, train_mask: np.ndarray,
            variables: tuple[str, ...] = C.ALL_VARS) -> "StandardScalerPerStation":
        tr = panel.loc[train_mask]
        mean, std = {}, {}
        for st in C.STATIONS:
            sub = tr[tr.site_id == st]
            for var in variables:
                series = np.log1p(sub[var]) if var in C.LOG1P_VARS else sub[var]
                mean[(st, var)] = float(series.mean())
                std[(st, var)] = float(series.std() + 1e-8)
        return cls(mean=mean, std=std)

    def transform_value(self, station: str, var: str, x: np.ndarray) -> np.ndarray:
        if var in C.LOG1P_VARS:
            x = np.log1p(np.clip(x, 0, None))
        return (x - self.mean[(station, var)]) / self.std[(station, var)]


def prepare_dataset() -> dict[str, object]:
    """One-stop builder for the 3-station cascade: panel + masks + imputer."""
    assert_split_disjoint()
    panel = load_panel()
    masks = split_masks(panel["DATE"])
    train_mask = masks.train
    imputer = Imputer.fit(panel, train_mask)
    panel_imp = imputer.transform(panel)
    return {"panel_raw": panel, "panel": panel_imp, "masks": masks, "imputer": imputer}


def prepare_dataset_from_panel(panel_path: str, set_global_stations: bool = True) -> dict[str, object]:
    """Same fold-safe pipeline applied to an externally-acquired panel
    (e.g. ``data_usgs/panel_usgs_100.parquet``).

    Reads the panel, registers its station list as ``C.STATIONS`` (so the rest of
    the code — scalers, climatology, ThermoRoute n_stations — sees the right
    dimension), masks observed flags, builds time-split masks, fits the imputer
    on the training fold and returns the imputed panel.

    The 3-station ``prepare_dataset`` and 09's previous private ``prep()`` are
    superseded by this single entry point; both 3-station and USGS pipelines now
    share one fold-safe preparation step.
    """
    panel = pd.read_parquet(panel_path)
    panel["DATE"] = pd.to_datetime(panel["DATE"])
    stations = tuple(sorted(panel.site_id.unique()))
    if set_global_stations:
        C.STATIONS = stations
        C.UPSTREAM = {s: None for s in stations}
    for v in C.ALL_VARS:
        if v in panel.columns:
            panel[f"{v}_observed"] = panel[v].notna()
    masks = split_masks(panel["DATE"])
    imputer = Imputer.fit(panel, masks.train)
    panel_imp = imputer.transform(panel)
    return {
        "panel_raw": panel, "panel": panel_imp, "masks": masks,
        "imputer": imputer, "stations": stations,
    }
