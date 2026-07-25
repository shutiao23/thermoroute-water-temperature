from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.adaptive import delayed_aci


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "22_adaptive_conformal.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "thermoroute_test_adaptive_conformal",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delayed_aci_does_not_use_future_feedback():
    evaluation = pd.DataFrame({
        "issue_date": pd.date_range("2020-01-01", periods=10),
        "target_date": pd.date_range("2020-01-08", periods=10),
        "y_true": 1.0,
        "q05": 0.0,
        "q95": 0.1,
    })
    output = delayed_aci(np.linspace(0, 1, 100), evaluation, alpha=0.1, gamma=0.02)
    assert output.feedback_count.iloc[:7].eq(0).all()
    assert output.feedback_count.iloc[7] == 1


def test_delayed_aci_reports_width_and_interval_score():
    evaluation = pd.DataFrame({
        "issue_date": pd.date_range("2020-01-01", periods=5),
        "target_date": pd.date_range("2020-01-02", periods=5),
        "y_true": [0.0, 0.2, 0.4, 0.6, 0.8],
        "q05": 0.0,
        "q95": 0.5,
    })
    output = delayed_aci(np.linspace(0, 0.2, 100), evaluation)
    assert output.aci_width.notna().all()
    assert output.aci_interval_score.notna().all()
    assert (output.aci_interval_score >= output.aci_width).all()


def test_adaptive_conformal_source_rejects_calendar_day_and_real_arrival_names():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "7-retained-row block-max CQR sensitivity" in source
    assert "idealized target-date-arrival delayed ACI" in source
    assert "block_days=BLOCK_RETAINED_ROWS" in source
    assert "seven-day block-CQR" not in source
    assert "7-day block-CQR" not in source
    assert "delayed-feedback ACI" not in source


def test_adaptive_conformal_report_states_exact_sensitivity_boundaries():
    module = _load_script()
    horizons = list(module.C.HORIZONS)
    result = pd.DataFrame({
        "horizon": horizons,
        "warm": [index == 0 for index in range(len(horizons))],
        "split_covered": True,
        "split_width": 1.0,
        "split_interval_score": 1.0,
        "block_covered": True,
        "block_width": 1.1,
        "block_interval_score": 1.1,
    })
    for gamma in module.GAMMAS:
        tag = f"aci_{str(gamma).replace('.', 'p')}"
        result[f"{tag}_covered"] = True
        result[f"{tag}_width"] = 1.2
        result[f"{tag}_interval_score"] = 1.2

    report = module._render_report(result)

    assert "7-retained-row block-max CQR sensitivity" in report
    assert "7 retained rows" in report
    assert "not seven-calendar-day blocks" in report
    assert "idealized target-date-arrival delayed ACI" in report
    assert "no verified observation-publication timestamps" in report
    assert "source revision or vintage histories" in report
    assert "reporting-latency records" in report
    assert "neither observes nor replays real feedback availability" in report
    assert "do not estimate conditional coverage" in report
    assert "7-day block-CQR" not in report
    assert "\n| delayed ACI gamma=" not in report
