from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import opening as OPENING  # noqa: E402


class _FixedHead:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, _design, num_threads=1):
        assert num_threads == 1
        return self.values.copy()


def test_trusted_lightgbm_scoring_preserves_nominal_q50(monkeypatch, tmp_path):
    raw_q50 = np.array([2.0, 1.0])
    heads = {
        "point": _FixedHead([1.5, 1.5]),
        "q05": _FixedHead([4.0, 0.0]),
        "q50": _FixedHead(raw_q50),
        "q95": _FixedHead([1.0, 3.0]),
        "event": _FixedHead([0.2, 0.8]),
    }
    members = {f"seed{seed}": {1: heads} for seed in range(5)}
    manifest = {
        "raw_feature_order": ["WTEMP"],
        "conformal_offsets": {"fixture-site|1": 0.0},
        "event_calibrators": {
            "1": {"intercept": 0.0, "slope": 1.0, "constant": 0.5}
        },
        "event_thresholds": {"fixture-site": 20.0},
    }
    monkeypatch.setattr(
        OPENING, "_verify_file_binding", lambda *_args, **_kwargs: tmp_path / "x"
    )
    monkeypatch.setattr(
        OPENING, "load_lightgbm_bundle", lambda _path: (members, manifest)
    )
    monkeypatch.setattr(
        OPENING,
        "_confirmation_tabular_design",
        lambda *_args, **_kwargs: pd.DataFrame({"x": [0.0, 1.0]}),
    )
    issue_dates = np.array(["2019-01-01", "2019-01-02"], dtype="datetime64[D]")
    wd = SimpleNamespace(
        y=np.array([[10.0], [11.0]]),
        target_valid=np.ones((2, 1), dtype=bool),
        horizons=(1,),
        station=np.zeros(2, dtype=int),
        issue_date=issue_dates,
        target_date=(issue_dates + np.timedelta64(1, "D"))[:, None],
    )
    scored = OPENING._score_lightgbm_bundle(
        root=tmp_path,
        entry={"artifact": {}},
        manifest=manifest,
        wd=wd,
        imputed=pd.DataFrame(),
        climatology=object(),
        expected=pd.DataFrame(),
        station_order=("fixture-site",),
        cohort="temporal",
        external=False,
    )
    assert np.array_equal(scored["q50"].to_numpy(float), raw_q50)
    assert np.array_equal(scored["q05"].to_numpy(float), np.array([2.0, 0.0]))
    assert np.array_equal(scored["q95"].to_numpy(float), np.array([2.0, 3.0]))
