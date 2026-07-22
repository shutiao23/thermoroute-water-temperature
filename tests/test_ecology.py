from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.ecology import active_standard_mask, compute_7dadm, validate_standard_registry


def test_7dadm_uses_daily_maximum_and_requires_seven_consecutive_days():
    data = pd.DataFrame({
        "DATE": pd.date_range("2020-01-01", periods=9, freq="D"),
        "site_no": "01234567",
        "WTEMP_MAX": np.arange(1.0, 10.0),
    })
    data.loc[3, "WTEMP_MAX"] = np.nan
    output = compute_7dadm(data)
    assert output.SEVEN_DADM.iloc[:9].isna().all()  # every seven-day window includes the gap

    complete = data.assign(WTEMP_MAX=np.arange(1.0, 10.0))
    output = compute_7dadm(complete)
    assert np.isclose(output.SEVEN_DADM.iloc[6], 4.0)
    assert np.isclose(output.SEVEN_DADM.iloc[8], 6.0)


def test_7dadm_refuses_daily_mean_target():
    frame = pd.DataFrame({"DATE": ["2020-01-01"], "site_no": ["1"], "WTEMP": [10]})
    with pytest.raises(ValueError, match="daily-maximum"):
        compute_7dadm(frame, maximum_col="WTEMP")


def test_standard_registry_requires_context_and_source():
    valid = pd.DataFrame({
        "site_no": ["01234567"],
        "jurisdiction": ["example"],
        "designated_use": ["cold-water habitat"],
        "species_life_stage": ["rearing"],
        "season_start": ["06-01"],
        "season_end": ["09-30"],
        "threshold_c": [18.0],
        "source_url": ["https://example.org/standard"],
    })
    validate_standard_registry(valid)
    with pytest.raises(ValueError, match="source_url"):
        validate_standard_registry(valid.drop(columns="source_url"))


def test_cross_year_standard_season():
    dates = pd.to_datetime(["2020-12-15", "2021-02-01", "2021-07-01"])
    assert active_standard_mask(dates, "11-01", "03-31").tolist() == [True, True, False]
