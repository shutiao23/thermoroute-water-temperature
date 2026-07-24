"""Regression checks for the three ordinary legacy monitoring sites."""

from __future__ import annotations

from pathlib import Path
import re
import runpy
import sys

import pandas as pd


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
        ROOT / ".github/workflows/ci.yml",
        ROOT / "README.md",
        ROOT / "scripts/01_prepare_data.py",
        ROOT / "scripts/06_make_figures.py",
        ROOT / "scripts/run_all.sh",
        ROOT / "outputs/README.md",
        ROOT / "paper/ThermoRoute_paper.md",
        ROOT / "paper/agu_submission/ThermoRoute_WRR.tex",
        ROOT / "src/thermoroute/__init__.py",
        ROOT / "src/thermoroute/config.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        r"b1\s*(?:→|->)\s*s2",
        r"s2\s*(?:→|->)\s*p3",
        r"confirms?\s+b1",
        r"station topology\s*\(directed cascade",
        r"near[- ]deterministic\s+cascade",
        r'"cascade_(?:predictions|scores)"',
        r"三站级联",
        r"级联(?:实验|主表|\s*Track A|的\s*PICP|决策价值)",
        r"b1\s+is\s+more\s+regulated",
    )
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None, pattern


def test_stale_adversarial_reports_are_not_tracked_as_current_evidence() -> None:
    assert not (ROOT / "outputs/reports/adversarial_review_tri_persona.md").exists()
    assert not (ROOT / "outputs/reports/adversarial_review_strong_accept.md").exists()
    note = (ROOT / "outputs/README.md").read_text(encoding="utf-8")
    assert "Historical files formerly stored" in note
    assert "ordinary monitoring sites" in note
    assert "not an authority document" in note


def test_canonical_run_does_not_silently_execute_the_legacy_case() -> None:
    script = (ROOT / "scripts/run_all.sh").read_text(encoding="utf-8")
    assert "INCLUDE_LEGACY_MONITORING_CASE=0" in script
    assert "--include-legacy-monitoring-case" in script
    assert "OPTIONAL LEGACY MONITORING-SITE CASE" in script
    assert "ROUTE A: USGS DEVELOPMENT ANALYSIS" in script
    assert "TRACK B" not in script


def test_legacy_site_figure_has_no_network_or_travel_time_semantics() -> None:
    namespace = runpy.run_path(str(ROOT / "scripts/06_make_figures.py"))
    figure_function = namespace["fig_study_area"]
    captured: dict[str, object] = {}

    def capture(figure: object, name: str) -> None:
        captured["figure"] = figure
        captured["name"] = name

    figure_function.__globals__["_save"] = capture
    panel = pd.DataFrame(
        {
            "site_id": ["b1", "b1", "s2", "s2", "p3", "p3"],
            "WTEMP": [4.0, 12.0, 6.0, 14.0, 8.0, 16.0],
        }
    )
    figure_function(panel)

    assert captured["name"] == "fig1_study_area"
    figure = captured["figure"]
    axes = figure.axes  # type: ignore[attr-defined]
    rendered_text = "\n".join(
        text.get_text() for axis in axes for text in axis.texts
    )
    assert rendered_text.count("monitoring\nsite") == 3
    assert "Display order only" in rendered_text
    assert "No reservoir, hydraulic-connectivity, or travel-time claim" in rendered_text
    for forbidden in (
        "Reservoir cascade",
        "flow ~",
        "thermal ~",
        "upstream",
        "downstream",
        "2480 m",
        "1819 m",
        "989 m",
    ):
        assert forbidden not in rendered_text
    namespace["plt"].close(figure)
