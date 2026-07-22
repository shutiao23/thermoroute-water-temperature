"""Outcome-free environmental scope audit for the canonical development cohort."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .provenance import canonical_json_bytes, sha256_file


AUDIT_FORMAT = "thermoroute.development-environmental-audit.v1"
FIELDS = ("WTEMP", "FLOW", "WLEVEL", "TEMP", "PRCP", "RHMEAN", "DH", "WDSP")


class EnvironmentalAuditError(RuntimeError):
    """The pre-opening environmental audit cannot be computed safely."""


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _finite_summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return {"finite_count": 0, "min": None, "median": None, "max": None}
    return {
        "finite_count": int(len(numeric)),
        "min": float(np.min(numeric)),
        "median": float(np.median(numeric)),
        "max": float(np.max(numeric)),
    }


def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    latitude = np.radians(lat)
    longitude = np.radians(lon)
    delta_lat = latitude[:, None] - latitude[None, :]
    delta_lon = longitude[:, None] - longitude[None, :]
    value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(latitude[:, None])
        * np.cos(latitude[None, :])
        * np.sin(delta_lon / 2.0) ** 2
    )
    distance = 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))
    np.fill_diagonal(distance, np.inf)
    return distance


def _split(date: pd.Series) -> pd.Series:
    year = pd.to_datetime(date).dt.year
    return pd.Series(
        np.select(
            [year <= 2015, year <= 2017, year == 2018, year <= 2020],
            ["train_2006_2015", "validation_2016_2017", "calibration_2018", "development_2019_2020"],
            default="outside_contract",
        ),
        index=date.index,
        dtype="string",
    )


def audit_development_environment(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    rejected: pd.DataFrame,
) -> dict[str, Any]:
    """Compute deterministic cohort, calendar, missingness and spatial diagnostics."""
    required_panel = {"site_id", "DATE", *FIELDS}
    required_registry = {
        "site_no", "legacy_site_id", "state", "huc_cd", "huc2", "lat", "lon",
        "drain_area_va", "station_nm", "wtemp_cov", "flow_cov", "wtemp_cov_test",
        "flow_cov_test",
    }
    if required_panel - set(panel) or required_registry - set(registry):
        raise EnvironmentalAuditError("panel or registry lacks audit fields")
    value = panel[["site_id", "DATE", *FIELDS]].copy()
    value["site_id"] = value.site_id.astype("string")
    value["DATE"] = pd.to_datetime(value.DATE, errors="coerce").dt.normalize()
    if value.DATE.isna().any() or value.duplicated(["site_id", "DATE"]).any():
        raise EnvironmentalAuditError("panel daily key is invalid or duplicated")
    if value.DATE.min() != pd.Timestamp("2006-01-01") or value.DATE.max() != pd.Timestamp(
        "2020-12-31"
    ):
        raise EnvironmentalAuditError("audit may read only the frozen 2006-2020 panel")
    expected_dates = pd.date_range("2006-01-01", "2020-12-31", freq="D")
    sites = tuple(sorted(value.site_id.astype(str).unique()))
    if len(sites) != 120 or len(value) != 120 * len(expected_dates):
        raise EnvironmentalAuditError("canonical panel shape changed")
    for site, group in value.groupby("site_id", sort=False):
        if not pd.DatetimeIndex(group.DATE.sort_values()).equals(expected_dates):
            raise EnvironmentalAuditError(f"panel calendar is not daily-complete: {site}")

    registry = registry.copy()
    registry["site_no"] = registry.site_no.astype("string")
    registry["legacy_site_id"] = registry.legacy_site_id.astype("string")
    if len(registry) != 120 or registry.site_no.duplicated().any():
        raise EnvironmentalAuditError("stable station registry changed")
    if set(registry.legacy_site_id.astype(str)) != set(sites):
        raise EnvironmentalAuditError("panel and station registry identities differ")
    value["split"] = _split(value.DATE)
    if value.split.eq("outside_contract").any():
        raise EnvironmentalAuditError("panel contains a date outside frozen temporal roles")

    missing_by_split: dict[str, Any] = {}
    for split, frame in value.groupby("split", sort=False):
        missing_by_split[str(split)] = {
            "row_count": int(len(frame)),
            "observed_count": {field: int(frame[field].notna().sum()) for field in FIELDS},
            "missing_fraction": {
                field: float(frame[field].isna().mean()) for field in FIELDS
            },
        }
    longest: dict[str, Any] = {}
    for field in FIELDS:
        per_site = {
            str(site): _longest_true_run(group[field].isna().to_numpy())
            for site, group in value.groupby("site_id", sort=False)
        }
        worst_site = max(per_site, key=lambda site: per_site[site])
        longest[field] = {
            "maximum_days": int(per_site[worst_site]),
            "worst_legacy_site_id": worst_site,
            "stations_with_any_missing": int(sum(days > 0 for days in per_site.values())),
        }

    lat = pd.to_numeric(registry.lat, errors="coerce").to_numpy(float)
    lon = pd.to_numeric(registry.lon, errors="coerce").to_numpy(float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise EnvironmentalAuditError("registry coordinates are invalid")
    nearest = np.min(_haversine_km(lat, lon), axis=1)
    huc = registry.huc_cd.astype("string").str.strip()
    station_names = registry.station_nm.astype("string").str.lower()
    regulation_pattern = r"\b(?:dam|reservoir|diversion|tailrace|below)\b"
    negative_flow = pd.to_numeric(value.FLOW, errors="coerce") < 0.0

    rejected = rejected.copy()
    if "reason" not in rejected:
        raise EnvironmentalAuditError("legacy rejection ledger lacks reason")
    # The immutable legacy ledger used the misleading phrase below.  Preserve
    # its bytes for provenance but correct the wording in every derived audit.
    corrected_reason = rejected.reason.astype("string").replace({
        "low blind-test-period coverage": (
            "low 2019-2020 development-evaluation-period coverage"
        )
    })
    return {
        "format": AUDIT_FORMAT,
        "status": "DESCRIPTIVE_PREOPEN_AUDIT_NO_POST_2020_DATA",
        "post_2020_values_read": False,
        "panel": {
            "row_count": int(len(value)),
            "station_count": int(len(sites)),
            "start": value.DATE.min().strftime("%Y-%m-%d"),
            "end": value.DATE.max().strftime("%Y-%m-%d"),
            "daily_calendar_complete_for_every_station": True,
        },
        "selection": {
            "retained_station_count": int(len(registry)),
            "recorded_rejection_count": int(len(rejected)),
            "recorded_rejection_reason_counts_corrected": {
                str(key): int(count)
                for key, count in corrected_reason.value_counts().sort_index().items()
            },
            "retained_coverage": {
                field: _finite_summary(registry[field])
                for field in ("wtemp_cov", "flow_cov", "wtemp_cov_test", "flow_cov_test")
            },
            "rejected_coverage": {
                field: _finite_summary(rejected[field])
                for field in ("wtemp_cov", "flow_cov", "wtemp_cov_test", "flow_cov_test")
                if field in rejected
            },
            "interpretation": (
                "Availability-enriched convenience cohort, not a probability sample or a "
                "nationally representative sample of U.S. rivers."
            ),
        },
        "geography": {
            "state_count": int(registry.state.astype("string").nunique()),
            "state_station_counts": {
                str(key): int(count)
                for key, count in registry.state.astype("string").value_counts().sort_index().items()
            },
            "huc2_count": int(registry.huc2.nunique()),
            "huc2_station_counts": {
                str(key): int(count)
                for key, count in registry.huc2.astype("string").value_counts().sort_index().items()
            },
            "unique_huc_code_count": int(huc.nunique()),
            "stations_in_repeated_huc_code_groups": int(huc.duplicated(keep=False).sum()),
            "nearest_station_distance_km": {
                "minimum": float(np.min(nearest)),
                "median": float(np.median(nearest)),
                "stations_with_neighbor_within_10km": int((nearest <= 10.0).sum()),
            },
            "drainage_area": _finite_summary(registry.drain_area_va),
            "station_name_regulation_keyword_count": int(
                station_names.str.contains(regulation_pattern, regex=True, na=False).sum()
            ),
            "regulation_keyword_is_only_a_heuristic": True,
        },
        "data_quality": {
            "missingness_by_split": missing_by_split,
            "longest_missing_run_by_field": longest,
            "measurement_provenance_limit": {
                "nwis_qualifier_columns_retained": False,
                "nwis_method_or_sensor_history_retained": False,
                "original_development_nwis_http_responses_retained": False,
                "interpretation": (
                    "The frozen development panel preserves daily values and missingness, "
                    "but it cannot support qualifier-, method-, or sensor-continuity "
                    "sensitivity analyses and does not reconstruct missing raw provenance."
                ),
            },
            "negative_flow": {
                "row_count": int(negative_flow.sum()),
                "station_count": int(value.loc[negative_flow, "site_id"].nunique()),
                "minimum_cfs": float(pd.to_numeric(value.FLOW, errors="coerce").min()),
                "interpretation": (
                    "Signed values are preserved numerically; they may reflect tidal/backwater "
                    "or measurement semantics and cannot automatically be called drought."
                ),
            },
            "wlevel_missing_fraction": float(value.WLEVEL.isna().mean()),
        },
        "fixed_scope_limits": [
            "HUC2 is a coarse grouping, not an independent river-network component.",
            "Nearby or repeated-HUC stations are not independent reaches.",
            "One 2021-2023 target interval cannot establish long-term climate generalization.",
            "Station-coordinate meteorology is not an upstream catchment average.",
            "The legacy development panel lacks NWIS qualifier, method, sensor-history, "
            "and original-response provenance needed to audit measurement discontinuities.",
            "Daily mean WTEMP cannot establish daily-maximum, 7DADM, species, or regulatory risk.",
        ],
    }


def render_environmental_audit_markdown(document: dict[str, Any]) -> str:
    """Render a concise human-readable companion without adding new claims."""
    selection = document["selection"]
    geography = document["geography"]
    quality = document["data_quality"]
    lines = [
        "# Development environmental-scope audit",
        "",
        f"Status: **{document['status']}**.",
        "",
        "This report uses only the frozen 2006–2020 development panel and metadata.",
        "It does not inspect any target-period outcome.",
        "",
        "## Cohort and geography",
        "",
        f"- Retained stations: {selection['retained_station_count']}",
        f"- Recorded rejected candidates: {selection['recorded_rejection_count']}",
        f"- States: {geography['state_count']}",
        f"- HUC2 groups: {geography['huc2_count']}",
        f"- Stations in repeated HUC-code groups: {geography['stations_in_repeated_huc_code_groups']}",
        "- Stations with another retained station within 10 km: "
        f"{geography['nearest_station_distance_km']['stations_with_neighbor_within_10km']}",
        "",
        "The cohort is availability-enriched and is not a national probability sample.",
        "",
        "## Data-quality facts",
        "",
        f"- Negative FLOW rows: {quality['negative_flow']['row_count']} across "
        f"{quality['negative_flow']['station_count']} stations",
        f"- Minimum FLOW: {quality['negative_flow']['minimum_cfs']:.3f} cfs",
        f"- WLEVEL missing fraction: {quality['wlevel_missing_fraction']:.4f}",
        "- Legacy NWIS qualifiers, method/sensor histories, and original HTTP responses: "
        "not retained",
        "",
        "## Permanent interpretation limits",
        "",
        *[f"- {value}" for value in document["fixed_scope_limits"]],
        "",
    ]
    return "\n".join(lines)


def write_environmental_audit(
    *,
    panel_path: str | Path,
    registry_path: str | Path,
    rejected_path: str | Path,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    panel_path = Path(panel_path).resolve()
    registry_path = Path(registry_path).resolve()
    rejected_path = Path(rejected_path).resolve()
    json_path = Path(json_path).resolve()
    markdown_path = Path(markdown_path).resolve()
    if json_path.exists() or markdown_path.exists():
        raise EnvironmentalAuditError("refusing to replace environmental audit")
    panel = pd.read_parquet(panel_path)
    registry = pd.read_csv(registry_path, dtype={"site_no": "string"})
    rejected = pd.read_csv(rejected_path, dtype={"site": "string"})
    document = audit_development_environment(panel, registry, rejected)
    document["inputs"] = {
        "panel": {"path": panel_path.name, "sha256": sha256_file(panel_path)},
        "registry": {"path": registry_path.name, "sha256": sha256_file(registry_path)},
        "rejected": {"path": rejected_path.name, "sha256": sha256_file(rejected_path)},
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (json_path, canonical_json_bytes(document)),
        (markdown_path, render_environmental_audit_markdown(document).encode("utf-8")),
    ):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return document
