"""Injection battery for the consistency gate (regression tests).

The gate carried defects twice (the findall capture-group bug and the
any-instead-of-all semantics) precisely because it had no tests of its own.
These tests copy the manuscript to a temporary file, inject one defect class
at a time, and require the gate to FAIL.  Baseline (no injection) must PASS.

Defect classes covered:
    INJ2  generated-table internal row drift (full-block check)
    INJ4  prose drift in a single used_in span (all-mode semantics)
    INJ5  prose drift in every span (all-mode semantics, whole document)
    INJ6  TeX-only drift (tex/md cross-check)
    INJ7  Conclusions-only drift of a four-span claim
    INJ8  Key-Points-only drift of a four-span claim
    INJ9  SI07 reportable value drift (SI file scanning)
    INJ10 SI07 unfiltered sensitivity value drift (SI file scanning)
"""

import sys

import pytest

sys.path.insert(0, "scripts/final")
import check_manuscript_consistency as G

REAL_MANUSCRIPT = G.MANUSCRIPT.read_text(encoding="utf-8")


@pytest.fixture
def gate_manuscript(tmp_path, monkeypatch):
    """A writable copy of the manuscript; the gate reads it via G.MANUSCRIPT."""
    copied = tmp_path / "ThermoRoute_paper.md"
    copied.write_text(REAL_MANUSCRIPT, encoding="utf-8")
    monkeypatch.setattr(G, "MANUSCRIPT", copied)
    return copied


def run_gate() -> int:
    # in-process: the test's monkeypatches on the G module apply
    return G.main()


def inject_span(manuscript, heading: str, needle: str, replacement: str,
                occurrence: int = 0) -> None:
    """Replace the ``occurrence``-th occurrence of ``needle`` after a heading."""
    text = manuscript.read_text(encoding="utf-8")
    idx = text.find(heading)
    assert idx >= 0, f"heading not found: {heading}"
    pos = idx
    for _ in range(occurrence + 1):
        pos = text.find(needle, pos)
        assert pos >= 0, f"needle not found after {heading}: {needle}"
        pos += 1
    pos -= 1
    text = text[:pos] + replacement + text[pos + len(needle):]
    manuscript.write_text(text, encoding="utf-8")


def test_baseline_gate_passes(gate_manuscript):
    assert run_gate() == 0


def test_inj2_generated_table_internal_row_drift(gate_manuscript):
    # Table 4.7a internal row: the old gate only checked each table's last row
    text = gate_manuscript.read_text(encoding="utf-8")
    assert "| LightGBM | 0.589 |" in text
    text = text.replace("| LightGBM | 0.589 |", "| LightGBM | 0.111 |", 1)
    gate_manuscript.write_text(text, encoding="utf-8")
    assert run_gate() == 1


def test_inj4_abstract_only_drift(gate_manuscript):
    # 0.589 is the one-day LightGBM RMSE, declared in both the Abstract and
    # Section 4.4; drifting only the Abstract copy must still fail the gate.
    inject_span(gate_manuscript, "## Abstract", "0.589", "0.999")
    assert run_gate() == 1


def test_inj5_whole_document_drift(gate_manuscript):
    text = gate_manuscript.read_text(encoding="utf-8")
    text = text.replace("1.694", "1.999")
    gate_manuscript.write_text(text, encoding="utf-8")
    assert run_gate() == 1


def test_inj6_tex_only_drift(tmp_path, monkeypatch):
    tex = tmp_path / "ThermoRoute_WRR.tex"
    tex.write_text(G.TEX.read_text(encoding="utf-8").replace(
        "1.694", "1.999"), encoding="utf-8")
    monkeypatch.setattr(G, "TEX", tex)
    assert run_gate() == 1


def test_inj7_conclusions_only_drift(gate_manuscript):
    inject_span(gate_manuscript, "## 7. Conclusions", "+0.250", "+0.999")
    assert run_gate() == 1


def test_inj8_key_points_only_drift(gate_manuscript):
    inject_span(gate_manuscript, "## Key Points", "+0.038", "+0.999")
    assert run_gate() == 1


def _si_root(tmp_path, monkeypatch, replacement: str):
    """tmp root whose paper/si holds a doctored SI07; everything else is
    symlinked from the real repo.  ``real_root`` is captured BEFORE the
    monkeypatch, so the symlink sources are the real files."""
    real_root = G.ROOT
    si_src = real_root / "paper" / "si" / "SI07_all_model_scores.md"
    si = tmp_path / "SI07_all_model_scores.md"
    si.write_text(si_src.read_text(encoding="utf-8").replace(
        "1.459", replacement).replace("1.478", replacement),
        encoding="utf-8")
    root = tmp_path / "root"
    monkeypatch.setattr(G, "ROOT", root)
    (root / "paper" / "si").mkdir(parents=True)
    import shutil
    shutil.copy(si, root / "paper" / "si" / si.name)
    for name in ("agu_submission", "ThermoRoute_paper.md", "tables_final.md"):
        src = real_root / "paper" / name
        dst = root / "paper" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            dst.symlink_to(src, target_is_directory=True)
        else:
            dst.symlink_to(src)
    (root / "outputs").symlink_to(G.FINAL, target_is_directory=True)
    (root / "paper" / "claim_ledger.yaml").symlink_to(
        real_root / "paper" / "claim_ledger.yaml")


def test_inj9_si07_reportable_drift(tmp_path, monkeypatch):
    _si_root(tmp_path, monkeypatch, "1.999")
    assert run_gate() == 1


def test_inj10_si07_unfiltered_drift(tmp_path, monkeypatch):
    _si_root(tmp_path, monkeypatch, "1.888")
    assert run_gate() == 1


def test_contradiction_scan_reports_stale_sibling():
    """The warning-level scan must fire on a stale sibling number."""
    import pandas as pd
    from check_manuscript_consistency import check_contradictions
    spans = {"abstract": "The seven-day RMSE is 1.478 C beside 1.459 C."}
    resolved = pd.DataFrame([{
        "claim_id": "T", "status": "RESOLVED", "value": "1.4589",
        "latex_macro": "X", "used_in": "abstract", "print_precision": None,
    }])
    warnings = check_contradictions(resolved, spans)
    assert any("1.478" in w and "T" in w for w in warnings)


# ---------------------------------------------------------------------------
# Phase-0 quarantine gates (claim status, withdrawn numbers, prohibited claims)
#
# INJ11  a provisional claim declared in a headline span
# INJ12  a provisional value printed in the Abstract without being declared
# INJ13  a claim with no status at all
# INJ14  a withdrawn headline number restored as an assertion
# INJ15  a protocol-forbidden operational claim about F3
# INJ16  a NOT_USED claim that still carries used_in spans
# ---------------------------------------------------------------------------


def test_inj11_provisional_claim_declared_in_a_headline_span(monkeypatch):
    ledger = [dict(claim) for claim in G.load_ledger()]
    for claim in ledger:
        if claim["claim_id"] == "RQ4_RAPID_WARMING_7D_DELTA":
            claim["used_in"] = ["abstract", "section_4_6"]
    problems = G.check_claim_status(ledger)
    assert any("DESCRIPTIVE_PROVISIONAL" in p and "abstract" in p for p in problems)


def test_inj12_provisional_value_printed_in_the_abstract(gate_manuscript):
    text = gate_manuscript.read_text(encoding="utf-8")
    marker = "**Plain Language Summary.**"
    assert marker in text
    leaked = text.replace(
        marker,
        "The learned model gains -0.300 °C on days that subsequently warm "
        "rapidly.\n\n" + marker,
        1,
    )
    gate_manuscript.write_text(leaked, encoding="utf-8")
    assert run_gate() == 1


def test_inj13_missing_claim_status():
    ledger = [dict(claim) for claim in G.load_ledger()]
    ledger[0].pop("status", None)
    problems = G.check_claim_status(ledger)
    assert any("expected one of" in p for p in problems)


def test_inj14_withdrawn_number_restored_as_an_assertion():
    problems = G.check_quarantined_content({
        "doc": "The whole-region geometry effect is +0.48 °C at seven days.",
    })
    assert any("withdrawn" in p for p in problems)


def test_inj14b_withdrawn_number_may_still_be_narrated():
    problems = G.check_quarantined_content({
        "doc": "DLOG-018 withdrew the +0.48 °C geometry effect and the 27:1 ratio.",
    })
    assert problems == []


def test_inj15_prohibited_operational_claim_about_f3():
    problems = G.check_quarantined_content({
        "doc": "Realized future meteorology yields an operational forecast gain "
               "of 0.63 degC at seven days.",
    })
    assert any("prohibited" in p for p in problems)


def test_inj15b_ratio_language_is_rejected():
    problems = G.check_quarantined_content({
        "doc": "Together these give the full information budget for the cohort.",
    })
    assert any("information budget" in p for p in problems)


def test_inj16_not_used_claim_still_carrying_spans():
    ledger = [{"claim_id": "X", "status": "NOT_USED", "used_in": ["abstract"]}]
    problems = G.check_claim_status(ledger)
    assert any("NOT_USED but still lists used_in" in p for p in problems)


def test_every_shipped_claim_declares_a_valid_status():
    assert G.check_claim_status(G.load_ledger()) == []
