#!/usr/bin/env python3
"""Check every shipped figure PDF for overlapping or undersized type.

``paper/figstyle.py`` already refuses to save a figure whose text artists
collide, which is why the main-text figures are clean. It only protects figures
drawn through it: the supporting-information figures come from a separate
skeleton renderer that never imported it, so nothing has ever checked them.

This checks the *artifacts* rather than the drawing code, so it covers every
figure regardless of which renderer produced it, and it keeps working if a
figure is ever hand-edited or dropped in from elsewhere. Word boxes come from
``pdftotext -bbox``, which reports the same coordinates the typesetter will use.

Two failures are reported, and they are the two a reader actually notices:

*Overlap* -- two words whose boxes intersect by more than a hair. Small
intersections are tolerated because kerned glyph boxes routinely touch;
``MIN_OVERLAP_PT`` is the threshold below which no reader could tell.

*Undersized type* -- text below AGU's floor at printed size. A figure drawn
larger than ``\\linewidth`` and then scaled down prints its type smaller than
the point size the renderer declared, which is the failure mode figstyle's
header describes: 7.5 pt drawn at 7.1 in and printed at 5.5 in arrives at
5.8 pt. Because the check reads the PDF's own media box, it sees the size the
figure will actually be placed at.

The size is estimated from glyph boxes, and a first version of this check got
that wrong in a way worth recording: it compared *box height* against a point
size. A word box is only as tall as its tallest glyph, so "7" reports its cap
height and "ace" its x-height, and the check flagged every axis tick in the
paper as undersized. Since a check that cries wolf is worse than no check, the
estimate now divides by the ratio appropriate to the glyphs actually present,
and only words containing an ascender or a capital -- where that ratio is
known rather than guessed -- are measured at all.
"""

from __future__ import annotations

import argparse
import re
import zlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIRS = (
    ROOT / "paper" / "agu_submission" / "figures",
    ROOT / "paper" / "si" / "figures",
)

#: AGU asks for roughly 8 pt minimum in figures.  figstyle draws this paper's
#: figures at 7-7.5 pt by design and at final size, so the floor here is set to
#: catch a figure that was drawn oversized and scaled down, not to relitigate
#: that choice.
MIN_POINT_SIZE = 6.0
#: Cap height as a fraction of point size for the sans faces in use.  Only
#: words containing a capital or an ascender are measured, so this ratio is the
#: right one; x-height-only words are skipped rather than guessed at.
CAP_HEIGHT_RATIO = 0.72
_ASCENDER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt0123456789")
#: Below this, two boxes are merely touching, which kerning does routinely.
MIN_OVERLAP_PT = 1.2
#: The width a figure is placed at in the AGU class, in points (139.7 mm).
LINEWIDTH_PT = 139.7 / 25.4 * 72.0

WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">'
    r"([^<]*)</word>"
)
PAGE_RE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">')


class TypographyError(RuntimeError):
    """A figure could not be read."""


def words(pdf: Path) -> tuple[list[dict[str, Any]], float]:
    result = subprocess.run(
        ["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise TypographyError(f"cannot read {pdf}: {result.stderr.strip()[:120]}")
    page = PAGE_RE.search(result.stdout)
    width = float(page.group(1)) if page else LINEWIDTH_PT
    found = []
    for match in WORD_RE.finditer(result.stdout):
        x0, y0, x1, y1 = (float(match.group(i)) for i in range(1, 5))
        text = match.group(5).strip()
        if text:
            found.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    return found, width


def overlaps(found: Sequence[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """Word pairs whose boxes genuinely intersect.

    Quadratic, which is fine: the largest figure here carries a few hundred
    words. Boxes are compared after shrinking each by half the tolerance, so a
    pair that merely abuts does not register.
    """
    hits = []
    pad = MIN_OVERLAP_PT / 2.0
    for i, a in enumerate(found):
        for b in found[i + 1:]:
            dx = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]) - pad
            dy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]) - pad
            if dx > 0 and dy > 0:
                hits.append((a["text"], b["text"], round(min(dx, dy), 2)))
    return hits


def undersized(found: Sequence[dict[str, Any]], width: float) -> list[tuple[str, float]]:
    """Words whose printed cap height falls below the floor.

    The scale factor is what matters and is easy to get wrong: a figure wider
    than the text column is shrunk by the class, taking its type with it.
    """
    scale = min(1.0, LINEWIDTH_PT / width) if width else 1.0
    small = []
    for word in found:
        if not (_ASCENDER & set(word["text"])):
            continue          # x-height only: the ratio would be a guess
        point_size = (word["y1"] - word["y0"]) * scale / CAP_HEIGHT_RATIO
        if point_size < MIN_POINT_SIZE:
            small.append((word["text"], round(point_size, 2)))
    return small


def declared_type_sizes(pdf: Path) -> list[float]:
    """Every type size the PDF actually sets, read from its ``Tf`` operators.

    The word-bbox measurement above cannot see a glyph smaller than the word it
    belongs to.  A mathtext exponent is set at 0.7x the base size, so a log tick
    reading "10^-3" at 7.5 pt puts 5.25 pt digits on the page inside a word whose
    bounding box measures 7.5 pt -- and the bbox check passes it. The shipped
    information-axes figure carried 5.25 pt type that way and this gate reported
    it clean, which is how it was found: by decompressing the content stream,
    not by the check that exists to find it.

    Reading the ``Tf`` operators is the direct measurement: it is what the
    renderer asked the typesetter for, per glyph run, with no inference from
    bounding boxes.
    """
    sizes: set[float] = set()
    data = pdf.read_bytes()
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            body = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        for size in re.findall(rb"/F\d+\s+([\d.]+)\s+Tf", body):
            sizes.add(round(float(size), 2))
    return sorted(sizes)


def check(pdf: Path) -> dict[str, Any]:
    found, width = words(pdf)
    collisions = overlaps(found)
    tiny = undersized(found, width)
    bad_text = [w["text"] for w in found
                if w["text"].lower() in {"nan", "+nan", "-nan", "inf", "none"}]
    declared = declared_type_sizes(pdf)
    return {
        "figure": pdf.stem, "words": len(found), "width_pt": round(width, 1),
        "overlaps": collisions, "undersized": tiny, "non_finite_labels": bad_text,
        "declared_sizes": declared,
        "declared_undersized": [x for x in declared if x < MIN_POINT_SIZE],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    problems = 0
    for directory in FIGURE_DIRS:
        for pdf in sorted(directory.glob("*.pdf")):
            if pdf.stem.startswith("agu-logo"):
                continue
            report = check(pdf)
            flags = []
            if report["overlaps"]:
                flags.append(f"{len(report['overlaps'])} overlapping word pairs")
            if report["undersized"]:
                flags.append(
                    f"{len(report['undersized'])} words under {MIN_POINT_SIZE} pt")
            if report["non_finite_labels"]:
                flags.append(
                    f"non-finite labels printed: {report['non_finite_labels']}")
            if report["declared_undersized"]:
                flags.append(
                    f"sets type at {report['declared_undersized']} pt, under the "
                    f"{MIN_POINT_SIZE} pt floor (usually a mathtext exponent: "
                    f"write the tick as a decimal instead of a power of ten)")
            if flags:
                problems += 1
                print(f"FAIL {report['figure']}: {'; '.join(flags)}")
                for a, b, amount in report["overlaps"][:4]:
                    print(f"       overlap {amount} pt: {a!r} x {b!r}")
                for text, size in report["undersized"][:4]:
                    print(f"       {size} pt: {text!r}")
            elif not args.quiet:
                print(f"ok   {report['figure']} ({report['words']} words, "
                      f"{report['width_pt']} pt wide)")
    print(f"\n{problems} figure(s) with typography problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
