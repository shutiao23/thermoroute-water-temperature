"""Strict NWIS acquisition re-parse for the conventional 2021-2023 holdout.

Re-parses the content-addressed snapshot cache with the frozen Route-A
confirmatory parser (``usgs.parse_nwis_confirmatory_daily``) so the panel is
built from exactly the same daily-mean-only semantics the sealed pipeline
used.  Each registry site receives a typed acquisition status:

* ``OK`` — at least one finite daily-mean series survives strict re-parse.
* ``NO_SERIES`` — the snapshot was acquired successfully but contains no
  finite daily-mean series (provably dry).
* ``ALL_SERIES_CONFLICT`` — multiple daily-mean series disagree on every day
  (the ``01435000`` case: 1,110 conflict days, 17 retained).
* ``PARSE_FAILED`` — the snapshot bytes do not satisfy the strict parser.
* ``HTTP_FAILED`` — the snapshot family records an HTTP failure (retryable).
* ``SNAPSHOT_MISSING`` — no snapshot exists for the requested URL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import SnapshotStore
from .usgs import NWIS_PARAMS, fetch_nwis_daily, parse_nwis_confirmatory_daily

ACQUISITION_STATUS_OK = "OK"
ACQUISITION_STATUS_NO_SERIES = "NO_SERIES"
ACQUISITION_STATUS_ALL_SERIES_CONFLICT = "ALL_SERIES_CONFLICT"
ACQUISITION_STATUS_PARSE_FAILED = "PARSE_FAILED"
ACQUISITION_STATUS_HTTP_FAILED = "HTTP_FAILED"
ACQUISITION_STATUS_SNAPSHOT_MISSING = "SNAPSHOT_MISSING"

NWIS_URL = (
    "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}"
    "&startDT={start}&endDT={end}&parameterCd={params}&siteStatus=all"
)


def nwis_snapshot_url(site: str, start: str, end: str) -> str:
    """The exact URL the 2021-2023 fetch used (cache key compatibility)."""
    from urllib.parse import urlencode

    return "https://waterservices.usgs.gov/nwis/dv/?" + urlencode({
        "format": "rdb",
        "sites": site,
        "startDT": start,
        "endDT": end,
        "parameterCd": ",".join(NWIS_PARAMS),
        "siteStatus": "all",
    })


def reparse_site(
    store: SnapshotStore,
    site: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Strictly re-parse one site's snapshot; return (panel_frame, status, detail)."""
    url = nwis_snapshot_url(site, start, end)
    try:
        payload, _record = store.fetch(
            provider="usgs-nwis-dv", url=url, retries=0
        )
    except Exception as exc:  # snapshot absent or unverified
        return pd.DataFrame(), ACQUISITION_STATUS_SNAPSHOT_MISSING, {"error": str(exc)[:200]}
    try:
        frame = parse_nwis_confirmatory_daily(payload, site_no=site, start=start, end=end)
    except Exception as exc:  # strict parser rejected the payload
        return pd.DataFrame(), ACQUISITION_STATUS_PARSE_FAILED, {"error": str(exc)[:200]}
    conflict_cols = [c for c in frame.columns if c.endswith("_series_conflict")]
    any_conflict = frame[conflict_cols].any(axis=1)
    wtemp = frame.get("WTEMP")
    if wtemp is None or not np.isfinite(wtemp.to_numpy(float)).any():
        if any_conflict.any():
            return frame, ACQUISITION_STATUS_ALL_SERIES_CONFLICT, {
                "conflict_days": int(any_conflict.sum()),
                "retained_days": int(np.isfinite(wtemp.to_numpy(float)).sum()) if wtemp is not None else 0,
            }
        return frame, ACQUISITION_STATUS_NO_SERIES, {"conflict_days": int(any_conflict.sum())}
    return frame, ACQUISITION_STATUS_OK, {"conflict_days": int(any_conflict.sum())}


def acquisition_cohort_table(
    registry: pd.DataFrame,
    store: SnapshotStore,
    start: str,
    end: str,
) -> pd.DataFrame:
    """One row per registry site: status, per-horizon target counts, reason."""
    rows = []
    for site in registry["site_no"].astype(str):
        frame, status, detail = reparse_site(store, site, start, end)
        n_finite = (
            int(np.isfinite(frame["WTEMP"].to_numpy(float)).sum())
            if not frame.empty else 0
        )
        rows.append({
            "site_no": site,
            "acquisition_status": status,
            "n_daily_finite_wtemp": n_finite,
            "conflict_days": detail.get("conflict_days", 0),
            "exclusion_reason": "" if status == ACQUISITION_STATUS_OK else status,
        })
    return pd.DataFrame(rows)


def retry_http_failed(
    sites: list[str],
    start: str,
    end: str,
    store: SnapshotStore,
) -> dict[str, str]:
    """Re-fetch sites whose snapshot family recorded an HTTP failure."""
    out: dict[str, str] = {}
    for site in sites:
        try:
            result = fetch_nwis_daily(site, start, end, snapshot_store=store)
        except Exception as exc:
            out[site] = f"retry raised: {str(exc)[:120]}"
            continue
        out[site] = "OK" if result is not None and "WTEMP" in result else "NO_SERIES"
    return out
