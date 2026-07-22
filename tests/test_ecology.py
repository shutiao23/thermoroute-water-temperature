from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.ecology import (
    active_standard_mask,
    apply_observed_standards,
    compute_7dadm,
    evaluate_observed_7dadm,
    validate_standard_registry,
)


@pytest.fixture
def observed_threshold_inputs():
    """Small observed fixture with equality, exceedance, and season boundaries."""
    maxima = pd.DataFrame({
        "DATE": pd.date_range("2020-05-28", periods=10, freq="D"),
        "site_no": "01234567",
        "WTEMP_MAX": [16.0] * 6 + [30.0] * 4,
    })
    standards = pd.DataFrame({
        "site_no": ["01234567"],
        "jurisdiction": ["Example State"],
        "designated_use": ["cold-water habitat"],
        "species_life_stage": ["rearing"],
        "season_start": ["06-03"],
        "season_end": ["06-04"],
        "threshold_c": [18.0],
        "source_url": ["https://example.org/adopted-standard"],
    })
    return maxima, standards


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


def test_observed_standard_fixture_applies_threshold_on_window_end_date(
    observed_threshold_inputs,
):
    maxima, standards = observed_threshold_inputs
    output = evaluate_observed_7dadm(maxima, standards).set_index("DATE")

    boundary = output.loc[pd.Timestamp("2020-06-03")]
    assert boundary.SEVEN_DADM == 18.0
    assert bool(boundary.applicable)
    assert not bool(boundary.exceedance)  # equality is not an exceedance
    assert boundary.comparison_status == "observed_at_or_below_threshold"

    exceedance = output.loc[pd.Timestamp("2020-06-04")]
    assert exceedance.SEVEN_DADM == 20.0
    assert bool(exceedance.exceedance)
    assert exceedance.comparison_status == "observed_above_threshold"

    outside = output.loc[pd.Timestamp("2020-06-05")]
    assert not bool(outside.applicable)
    assert pd.isna(outside.exceedance)
    assert outside.comparison_status == "outside_declared_season"


def test_standard_site_tamper_fails_closed(observed_threshold_inputs):
    maxima, standards = observed_threshold_inputs
    tampered = standards.assign(site_no="07654321")
    with pytest.raises(ValueError, match="site sets do not match exactly"):
        evaluate_observed_7dadm(maxima, tampered)


def test_derived_7dadm_tamper_is_rejected(observed_threshold_inputs):
    maxima, standards = observed_threshold_inputs
    seven = compute_7dadm(maxima)
    seven.loc[6, "SEVEN_DADM"] += 0.1
    with pytest.raises(ValueError, match="values do not match the daily maxima"):
        apply_observed_standards(seven, standards)


def test_ambiguous_standard_context_is_rejected(observed_threshold_inputs):
    _, standards = observed_threshold_inputs
    duplicate = pd.concat([
        standards,
        standards.assign(threshold_c=17.0, source_url="https://example.org/revision"),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="ambiguous duplicate"):
        validate_standard_registry(duplicate)

    overlap = pd.concat([
        standards.assign(season_start="05-01", season_end="06-15"),
        standards.assign(
            season_start="06-01",
            season_end="09-30",
            threshold_c=17.0,
            source_url="https://example.org/summer-standard",
        ),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="overlapping seasons"):
        validate_standard_registry(overlap)


def _load_threshold_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "21_ecological_thresholds.py"
    spec = importlib.util.spec_from_file_location("ecological_threshold_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage21_writes_applied_observed_descriptions_only(
    tmp_path, monkeypatch, observed_threshold_inputs,
):
    maxima, standards = observed_threshold_inputs
    maxima_path = tmp_path / "observed_daily_max.parquet"
    standards_path = tmp_path / "standards.csv"
    maxima.to_parquet(maxima_path, index=False)
    standards.to_csv(standards_path, index=False)
    module = _load_threshold_script()
    monkeypatch.setattr(module, "DAILY_MAXIMUM", maxima_path)
    monkeypatch.setattr(module, "STANDARDS", standards_path)
    monkeypatch.setattr(module.C, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(module.C, "REPORTS", tmp_path / "reports")

    module.main()

    output = pd.read_csv(tmp_path / "tables" / "eco_thresholds.csv")
    assert {"threshold_c", "applicable", "comparison_status", "exceedance"} <= set(output)
    assert not any("pred" in column.lower() for column in output.columns)
    assert output.loc[output.DATE == "2020-06-04", "exceedance"].item()
    report = (tmp_path / "reports" / "ecological_thresholds.md").read_text()
    assert "No model prediction is read or classified" in report
    assert "not a legal or regulatory compliance determination" in report


def test_stage21_missing_inputs_produces_no_classification(tmp_path, monkeypatch):
    module = _load_threshold_script()
    monkeypatch.setattr(module, "DAILY_MAXIMUM", tmp_path / "missing-maxima.parquet")
    monkeypatch.setattr(module, "STANDARDS", tmp_path / "missing-standards.csv")
    monkeypatch.setattr(module.C, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(module.C, "REPORTS", tmp_path / "reports")

    module.main()

    output = pd.read_csv(tmp_path / "tables" / "eco_thresholds.csv")
    assert output.empty
    report = (tmp_path / "reports" / "ecological_thresholds.md").read_text()
    assert "stage fails closed" in report.lower()
    assert "no observed exceedance classification" in report.lower()


def test_stage21_site_tamper_invalidates_stale_output(
    tmp_path, monkeypatch, observed_threshold_inputs,
):
    maxima, standards = observed_threshold_inputs
    maxima_path = tmp_path / "observed_daily_max.parquet"
    standards_path = tmp_path / "standards.csv"
    maxima.to_parquet(maxima_path, index=False)
    standards.assign(site_no="07654321").to_csv(standards_path, index=False)
    module = _load_threshold_script()
    monkeypatch.setattr(module, "DAILY_MAXIMUM", maxima_path)
    monkeypatch.setattr(module, "STANDARDS", standards_path)
    monkeypatch.setattr(module.C, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(module.C, "REPORTS", tmp_path / "reports")
    stale = tmp_path / "tables" / "eco_thresholds.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("exceedance\nTrue\n")

    with pytest.raises(ValueError, match="site sets do not match exactly"):
        module.main()

    assert pd.read_csv(stale).empty
    report = (tmp_path / "reports" / "ecological_thresholds.md").read_text()
    assert "input validation or threshold application was refused" in report
    assert "No observed exceedance classification" in report
