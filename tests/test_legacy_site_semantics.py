"""Regression checks for the three ordinary legacy monitoring sites."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import config as C  # noqa: E402


def test_legacy_monitoring_sites_do_not_encode_a_network() -> None:
    assert set(C.UPSTREAM) == {"b1", "s2", "p3"}
    assert all(upstream is None for upstream in C.UPSTREAM.values())
    assert C.FLOW_TRAVEL_DAYS == {}
    assert C.THERMAL_TRAVEL_DAYS == {}


def test_legacy_narratives_do_not_reintroduce_cascade_claims() -> None:
    paths = (
        ROOT / "scripts/01_prepare_data.py",
        ROOT / "outputs/reports/data_audit.md",
        ROOT / "paper/ThermoRoute_paper.md",
        ROOT / "paper/agu_submission/ThermoRoute_WRR.tex",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        r"b1\s*(?:→|->)\s*s2",
        r"s2\s*(?:→|->)\s*p3",
        r"confirms?\s+b1",
        r"station topology\s*\(directed cascade",
        r"b1\s+is\s+more\s+regulated",
    )
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern
