from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thermoroute.adaptive import delayed_aci


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
