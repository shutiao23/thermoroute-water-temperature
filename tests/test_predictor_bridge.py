from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute.predictor_bridge import (  # noqa: E402
    BRIDGE_FIELDS,
    PredictorBridgeError,
    compare_predictor_bridge,
    frozen_bridge_slice,
)


def _table() -> pd.DataFrame:
    dates = pd.date_range("2020-12-28", "2020-12-31", freq="D")
    rows = []
    for site_index, site in enumerate(("1", "2")):
        for day_index, date in enumerate(dates):
            base = float(10 * site_index + day_index)
            rows.append({
                "site_no": site,
                "DATE": date,
                "TEMP": base,
                "PRCP": base + 1,
                "RHMEAN": base + 2,
                "DH": np.nan if date == pd.Timestamp("2020-12-31") else base + 3,
                "WDSP": base + 4,
            })
    return pd.DataFrame(rows)


def test_exact_predictor_bridge_passes_and_attests_calendar():
    table = _table()
    report = compare_predictor_bridge(
        table,
        table.copy(),
        expected_site_count=2,
        start=pd.Timestamp("2020-12-28"),
        end=pd.Timestamp("2020-12-31"),
    )
    assert report["status"] == "PASS_EXACT_PRODUCT_BRIDGE"
    assert report["outcome_values_requested_or_read"] is False
    assert report["daymet_calendar_attestation"][
        "leap_year_omitted_dates_in_interval"
    ] == ["2020-12-31"]
    assert all(
        report["fields"][field]["exact_product_compatibility"]
        for field in BRIDGE_FIELDS
    )


def test_predictor_bridge_fails_product_drift_and_date_shift():
    frozen = _table()
    changed = frozen.copy()
    changed["TEMP"] = changed.groupby("site_no").TEMP.shift(1)
    report = compare_predictor_bridge(
        frozen,
        changed,
        expected_site_count=2,
        start=pd.Timestamp("2020-12-28"),
        end=pd.Timestamp("2020-12-31"),
    )
    assert report["status"] == "NO_GO_PRODUCT_BRIDGE_MISMATCH"
    assert report["fields"]["TEMP"]["exact_product_compatibility"] is False
    assert report["fields"]["TEMP"]["zero_day_alignment_best_or_tied"] is False


def test_predictor_bridge_rejects_incomplete_calendar():
    table = _table().iloc[:-1].copy()
    with pytest.raises(PredictorBridgeError, match="key registry is incomplete"):
        compare_predictor_bridge(
            table,
            table,
            expected_site_count=2,
            start=pd.Timestamp("2020-12-28"),
            end=pd.Timestamp("2020-12-31"),
        )


def test_frozen_bridge_slice_requires_one_to_one_stable_mapping():
    panel = _table().rename(columns={"site_no": "site_id"})
    registry = pd.DataFrame({
        "site_no": ["01000001", "01000002"],
        "legacy_site_id": ["1", "1"],
    })
    with pytest.raises(PredictorBridgeError, match="one-to-one"):
        frozen_bridge_slice(
            panel,
            registry,
            start=pd.Timestamp("2020-12-28"),
            end=pd.Timestamp("2020-12-31"),
        )


def test_legacy_rejection_wording_correction_is_byte_bound():
    correction = json.loads(
        (ROOT / "data_usgs/rejected_sites_120v2_corrections_v1.json")
        .read_text(encoding="utf-8")
    )
    source = ROOT / correction["source"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == correction["source"][
        "sha256"
    ]
    rows = pd.read_csv(source)
    legacy = correction["corrections"][0]
    assert int(rows.reason.eq(legacy["legacy_value"]).sum()) == legacy["row_count"]
    assert "development-evaluation-period" in legacy["corrected_interpretation"]
