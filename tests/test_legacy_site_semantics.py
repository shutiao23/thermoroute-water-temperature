"""Regression checks for the three ordinary legacy monitoring sites."""

from __future__ import annotations

import hashlib
import json
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
        ROOT / "scripts/07_make_tables.py",
        ROOT / "scripts/08_decision_value.py",
        ROOT / "scripts/run_all.sh",
        ROOT / "outputs/README.md",
        ROOT / "paper/ThermoRoute_paper.md",
        ROOT / "paper/cover_letter.md",
        ROOT / "paper/highlights.md",
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
        r"(?:b1|s2|p3)\s+(?:is|are|was|were)\s+(?:an?\s+)?"
        r"(?:reservoir|regulated|upstream|downstream)",
        r"(?:b1|s2|p3).{0,30}(?:feeds?|flows?\s+into|discharges?\s+to)"
        r".{0,30}(?:b1|s2|p3)",
        r"(?:b1|s2|p3)\s+(?:is\s+)?(?:upstream|downstream)\s+of\s+"
        r"(?:b1|s2|p3)",
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


def test_historical_three_site_notice_is_exactly_registered() -> None:
    relative = "protocols/legacy_three_site_semantics_notice_v1.md"
    notice_path = ROOT / relative
    notice = notice_path.read_text(encoding="utf-8")
    assert "ordinary monitoring stations, not reservoirs" in notice
    assert "no upstream/downstream" in notice
    assert "regulation status, or travel time" in notice
    assert "f0a09ba44296a896205e219984b95e8bfe728219" in notice
    assert "7c034e8976042899cbe2cc2a2f04322488010d84" in notice
    assert "08536ce771529b189c4a63a069168b1d0a78c490" in notice
    assert "withdrawn historical provenance, not results" in notice

    registry = json.loads(
        (ROOT / "protocols/route_a_claim_registry_v1.json").read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(notice_path.read_bytes()).hexdigest()
    assert relative in registry["documents"]
    assert relative in registry["required_documents"]
    assert registry["preopen_document_sha256"][relative] == digest
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[legacy three-site semantics notice]" in root_readme
    assert f"({relative})" in root_readme


def test_legacy_wlevel_report_has_only_raw_unknown_unit_qc_semantics() -> None:
    source = (ROOT / "scripts/01_prepare_data.py").read_text(encoding="utf-8")
    assert "Raw FLOW–WLEVEL association (QC only)" in source
    assert "raw legacy units" in source
    assert "units, and vertical datum are unverified" in source
    assert "rating curve" not in source.lower()
    assert "stage–discharge" not in source.lower()
    assert "WLEVEL drift 2006→2020 (m)" not in source


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
    assert "Unordered identifiers" in rendered_text
    assert "positions have no geographic or hydrologic meaning" in rendered_text
    assert "No reservoir, connectivity, regulation, or travel-time claim" in rendered_text
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

    left_axis = axes[0]
    assert not left_axis.lines
    assert all(
        patch.__class__.__name__ == "FancyBboxPatch"
        for patch in left_axis.patches
    )
    assert not {
        "FancyArrow",
        "FancyArrowPatch",
        "ConnectionPatch",
    } & {artist.__class__.__name__ for artist in left_axis.get_children()}
    site_positions = {
        text.get_text(): text.get_position()
        for text in left_axis.texts
        if text.get_text() in {"b1", "s2", "p3"}
    }
    assert set(site_positions) == {"b1", "s2", "p3"}
    assert [
        site for site, _ in sorted(site_positions.items(), key=lambda item: item[1][0])
    ] == sorted(site_positions)
    (x1, y1), (x2, y2), (x3, y3) = [
        site_positions[site] for site in sorted(site_positions)
    ]
    twice_triangle_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    assert abs(twice_triangle_area) > 1e-9
    namespace["plt"].close(figure)
