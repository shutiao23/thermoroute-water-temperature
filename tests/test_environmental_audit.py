from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.environmental_audit import (  # noqa: E402
    _longest_true_run,
    audit_development_environment,
)


def test_longest_missing_run_handles_boundaries():
    assert _longest_true_run([True, True, False, True]) == 2
    assert _longest_true_run([False, False]) == 0
    assert _longest_true_run([True, True, True]) == 3


def test_checked_in_environmental_audit_is_honest_and_deterministic():
    panel = pd.read_parquet(ROOT / "data_usgs/panel_usgs_120v2.parquet")
    registry = pd.read_csv(
        ROOT / "data_usgs/station_registry_v1.csv", dtype={"site_no": "string"}
    )
    rejected = pd.read_csv(
        ROOT / "data_usgs/rejected_sites_120v2.csv", dtype={"site": "string"}
    )
    audit = audit_development_environment(panel, registry, rejected)
    assert audit["post_2020_values_read"] is False
    assert audit["panel"] == {
        "row_count": 657480,
        "station_count": 120,
        "start": "2006-01-01",
        "end": "2020-12-31",
        "daily_calendar_complete_for_every_station": True,
    }
    assert audit["selection"]["recorded_rejection_count"] == 1345
    assert audit["geography"]["state_count"] == 34
    assert audit["geography"]["huc2_count"] == 15
    assert audit["data_quality"]["negative_flow"]["row_count"] == 2059
    assert audit["data_quality"]["negative_flow"]["station_count"] == 2
    assert audit["selection"]["interpretation"].startswith("Availability-enriched")
