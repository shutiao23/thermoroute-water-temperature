"""Regression tests for the AGU table fitter in ``paper/agu_submission/build_agu.py``.

Three defects let the formal-comparison table overprint itself in the compiled
PDF: the header row was excluded from the width measurement, every wrapped
column received an equal share of the leftover space, and the last column was
promoted regardless of its content.  Each is pinned here, together with the
numeric-sign protection that the fix made necessary.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD_AGU = ROOT / "paper" / "agu_submission" / "build_agu.py"


def _load_build_agu():
    spec = importlib.util.spec_from_file_location("build_agu", BUILD_AGU)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_agu = _load_build_agu()


# The five formal comparisons, as Pandoc emits them: ten columns whose second
# holds ``ThermoRoute vs. DampedPersistence`` and whose header carries the
# widest cell of every numeric column.
TABLE_4_6 = (
    r"\toprule\noalign{}" "\n"
    r"\# & Comparison & Lead & ΔRMSE (°C) & CI low & CI high & Win rate & "
    r"Stations & p (sign flip) & Holm p \\" "\n"
    r"\midrule\noalign{}" "\n"
    r"1 & ThermoRoute vs. DampedPersistence & 1 d & -0.129 & -0.199 & -0.076 "
    r"& 0.90 & 116 & 3.1e-05 & 1.5e-04 \\" "\n"
    r"4 & ThermoRoute vs. LightGBM & 3 d & 0.015 & 0.010 & 0.025 & 0.24 & 116 "
    r"& 1.0e+00 & 1.0e+00 \\" "\n"
)
TABLE_4_6_COLUMNS = "rllrrrrrrr"


_X_COLUMN_SPEC = re.compile(
    r">\{\\hsize=([0-9.]+)\\hsize\\(raggedright|raggedleft|centering)"
    r"\\arraybackslash\}X"
)


def _weights(spec: str) -> list[float]:
    return [float(value) for value, _ in _X_COLUMN_SPEC.findall(spec)]


def _wrapped_columns(spec: str) -> list[tuple[float, str]]:
    """``(hsize factor, alignment)`` for each wrapped column, in table order."""
    return [(float(value), align) for value, align in _X_COLUMN_SPEC.findall(spec)]


def test_header_row_is_measured() -> None:
    """The header is frequently the widest cell and must not be skipped.

    The row opens with an escaped ``\\#``, which an earlier ``startswith("\\\\")``
    guard treated as a LaTeX rule.  Skipping it made ``p (sign flip)`` look 7
    characters wide instead of 13.
    """
    widths, _ = build_agu._column_content_widths(TABLE_4_6_COLUMNS, TABLE_4_6)
    p_value_column = widths[8]
    assert p_value_column >= build_agu._text_points("p (sign flip)") - 1.0


def test_comparison_column_outweighs_its_numeric_neighbours() -> None:
    """Equal division is what starved the comparison column into overprinting."""
    spec, _ = build_agu._bounded_column_spec(TABLE_4_6_COLUMNS, TABLE_4_6)
    weights = _weights(spec)
    assert len(weights) >= 2, "expected several wrapped columns"
    assert max(weights) > 2 * min(weights)


def test_wrapped_columns_clear_their_longest_token() -> None:
    """``DampedPersistence`` cannot be hyphenated, so its column must hold it."""
    widths, tokens = build_agu._column_content_widths(
        TABLE_4_6_COLUMNS, TABLE_4_6
    )
    indices, separation = build_agu._table_layout(widths, tokens)
    assert 1 in indices, "the comparison column must wrap"
    budget = (
        build_agu.TABLE_WIDTH_POINTS - 2 * separation * len(widths)
    ) * build_agu.TABLE_WIDTH_SAFETY
    demand = sum(
        tokens[i] if i in indices else widths[i] for i in range(len(widths))
    )
    assert demand <= budget


def test_hsize_factors_sum_to_the_wrapped_column_count() -> None:
    """tabularx requires it; a wrong sum silently rescales the whole table."""
    spec, _ = build_agu._bounded_column_spec(TABLE_4_6_COLUMNS, TABLE_4_6)
    weights = _weights(spec)
    assert sum(weights) == pytest.approx(len(weights), abs=1e-6)


def test_promoted_column_keeps_pandoc_alignment() -> None:
    """A promoted numeric column must stay right-aligned, not silently flip."""
    spec, _ = build_agu._bounded_column_spec(TABLE_4_6_COLUMNS, TABLE_4_6)
    wrapped = _wrapped_columns(spec)
    widest = max(wrapped, key=lambda column: column[0])
    assert widest[1] == "raggedright", "the comparison column is Pandoc 'l'"
    assert any(
        align == "raggedleft" for _, align in wrapped
    ), "a promoted numeric column must keep its right alignment"


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("-0.129", r"\ensuremath{-}0.129"),
        ("-0.000", r"\ensuremath{-}0.000"),
        ("3.1e-05", r"3.1e\ensuremath{-}05"),
        ("1.0e+00", "1.0e+00"),
        ("1,060", "1,060"),
    ],
)
def test_numeric_signs_are_made_unbreakable(cell: str, expected: str) -> None:
    """TeX breaks after a hyphen; in a wrapped column that split ``-0.000``.

    The sign landed on one line and the digits on the next, which reads as a
    different number.
    """
    assert build_agu._protect_numeric_signs(cell) == expected


@pytest.mark.parametrize("dash_range", ["2006--2015", "2021--2023", "2016--2017"])
def test_en_dash_ranges_are_not_read_as_signs(dash_range: str) -> None:
    """``2006--2015`` is a date range, not ``2006`` minus ``2015``."""
    assert build_agu._protect_numeric_signs(dash_range) == dash_range


def test_math_minus_is_charged_its_own_width() -> None:
    """The cmsy minus is over twice the hyphen it replaces."""
    protected = build_agu._protect_numeric_signs("-0.129")
    assert build_agu._cell_points(protected) > build_agu._cell_points("0.129")


def test_generated_tex_has_no_unwrapped_wide_table() -> None:
    """Every tabularx in the checked-in TeX must carry ``\\hsize`` factors.

    A table Pandoc emitted with no wrapped column at all is the state that
    produced the original overflow.
    """
    tex = (ROOT / "paper" / "agu_submission" / "ThermoRoute_WRR.tex").read_text(
        encoding="utf-8"
    )
    specs = re.findall(r"\\begin\{tabularx\}\{[^}]*\}\{@\{\}(.*?)@\{\}\}", tex)
    assert specs, "no tabularx tables found in the generated TeX"
    for spec in specs:
        assert "X" in spec, f"tabularx with no X column: {spec}"
        assert r"\hsize=" in spec, f"unweighted X column: {spec}"
