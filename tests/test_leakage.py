"""Leakage and split-integrity tests — the checks reviewers care about most.

Run:  PYTHONPATH=src python3 -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from thermoroute import config as C
from thermoroute import data as D
from thermoroute import features as F
from thermoroute import datasets as DS


def _bundle():
    b = D.prepare_dataset()
    clim = F.HarmonicClimatology.fit(b["panel"], b["masks"].train)
    return b, clim


def test_split_disjoint_and_ordered():
    D.assert_split_disjoint()
    s = C.SPLIT.as_dict()
    assert s["train"][1] < s["val"][0] < s["val"][1] < s["calib"][0]
    assert s["calib"][1] < s["test"][0]


def test_no_calendar_gaps_or_dupes():
    panel = D.load_panel()
    for st in C.STATIONS:
        sub = panel[panel.site_id == st]
        full = pd.date_range(sub.DATE.min(), sub.DATE.max(), freq="D")
        assert len(full) == len(sub)
        assert sub.DATE.duplicated().sum() == 0


def test_sentinels_masked():
    panel = D.load_panel()
    assert panel["WDSP"].max() < 999.0
    assert panel["PRCP"].max() < 99.9


def test_imputation_only_uses_train():
    """An imputer fit on train must be reproducible from train rows alone."""
    panel = D.load_panel()
    masks = D.split_masks(panel.DATE)
    imp = D.Imputer.fit(panel, masks.train)
    # global medians equal the train-only medians (no future leak)
    tr = panel.loc[masks.train]
    for st in C.STATIONS:
        for v in C.ALL_VARS:
            ref = float(tr[tr.site_id == st][v].median())
            assert np.isclose(imp.global_median[(st, v)], ref, equal_nan=True)


def test_window_tail_equals_issue_value():
    """The last history step must invert to WTEMP_t — no future bleed."""
    b, clim = _bundle()
    wd = DS.build_windows(b["panel"], b["masks"], clim)
    DS._assert_no_leakage(wd, b["panel"])   # raises on any mismatch
    assert len(wd.X) > 10000


def test_target_is_strictly_future():
    """The stored target y[:, hi] must equal panel WTEMP at issue_date + h —
    verified against the panel itself on a subsample, not just h > 0."""
    b, clim = _bundle()
    wd = DS.build_windows(b["panel"], b["masks"], clim)
    lookup = {(s, d): w for s, d, w in zip(
        b["panel"].site_id, pd.to_datetime(b["panel"].DATE).to_numpy(),
        b["panel"].WTEMP.to_numpy(float))}
    rng = np.random.default_rng(0)
    sample = rng.choice(len(wd.X), size=min(500, len(wd.X)), replace=False)
    checked = 0
    for hi, h in enumerate(wd.horizons):
        assert h > 0
        for i in sample:
            st = C.STATIONS[wd.station[i]]
            tgt = lookup.get((st, wd.issue_date[i] + np.timedelta64(h, "D")))
            if tgt is not None and np.isfinite(tgt) and np.isfinite(wd.y[i, hi]):
                assert np.isclose(wd.y[i, hi], tgt, atol=1e-6), (
                    f"window {i}: y[h={h}]={wd.y[i, hi]} != panel WTEMP "
                    f"{tgt} at issue+{h}d for {st}")
                checked += 1
    assert checked > 500, "too few verifiable (window, horizon) pairs"
    # tabular path: target_date strictly after issue_date
    tab = F.attach_split(F.build_tabular(b["panel"], 3, C.FEATURE_SETS["V2"], clim))
    assert (pd.to_datetime(tab.target_date) > pd.to_datetime(tab.issue_date)).all()


def test_tabular_split_uses_issue_date():
    b, clim = _bundle()
    tab = F.attach_split(F.build_tabular(b["panel"], 7, C.FEATURE_SETS["V3"], clim))
    test_rows = tab[tab.split == "test"]
    assert pd.to_datetime(test_rows.issue_date).min() >= pd.Timestamp(C.SPLIT.test[0])
    train_rows = tab[tab.split == "train"]
    assert pd.to_datetime(train_rows.issue_date).max() <= pd.Timestamp(C.SPLIT.train[1])
