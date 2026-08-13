#!/usr/bin/env python3
"""Count the manuscript in AGU publication units, which is what AGU measures.

The submission was briefly cut to a 23-page PDF, and that was the wrong target.
AGU does not measure pages. It measures publication units:

    PU = words / 500 + figures + tables

and charges excess-length fees above 25. The word count is narrower than a
naive `wc -w` in ways that change the answer by thousands of words: it covers
the abstract, body text, in-text citations, figure captions and appendices, and
it *excludes* the title, the author list and affiliations, the plain-language
summary, the text inside tables, the Open Research section, the references, and
the entire Supporting Information.

Two consequences follow, and both were got wrong before this script existed.
References are free, so trimming the bibliography to save pages buys nothing
against the real limit. And page count is mostly a statement about line
spacing: the same manuscript is 23 pages single-spaced and 34 double-spaced,
and AGU asks for the double-spaced one. Optimising the page count therefore
traded a required format for a metric the publisher does not use.

Counting is deliberately conservative wherever the rule is ambiguous -- a
borderline inclusion is counted rather than dropped -- because the failure that
matters is discovering an excess-length fee after submission, not overstating
by a tenth of a unit.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper" / "ThermoRoute_paper.md"

#: Sections AGU excludes from the word count, as (start, end) markers. ``None``
#: as an end marker means "to the end of the document".
EXCLUDED_SPANS: tuple[tuple[str, str | None], ...] = (
    ("**Plain Language Summary.**", "**Keywords:**"),
    ("## 8. Open Research", "## Acknowledgments"),
    ("## Supporting Information", None),
)
#: Everything before the abstract is title, authors and affiliations.
BODY_STARTS_AT = "## Abstract"


def countable_words(markdown: str) -> tuple[int, dict[str, int]]:
    """Words AGU counts, with a breakdown of what was removed."""
    removed: dict[str, int] = {}
    text = markdown

    head, _, rest = text.partition(BODY_STARTS_AT)
    removed["title and author block"] = len(head.split())
    text = BODY_STARTS_AT + rest

    for start, end in EXCLUDED_SPANS:
        i = text.find(start)
        if i < 0:
            continue
        j = text.find(end, i + len(start)) if end else len(text)
        if j < 0:
            j = len(text)
        removed[start.strip("*# ").rstrip(".")] = len(text[i:j].split())
        text = text[:i] + text[j:]

    # Table interiors are excluded; captions are not, and captions live outside
    # the pipe block.
    interiors = re.findall(r"^\|.*$", text, flags=re.M)
    removed["table interiors"] = sum(len(row.split()) for row in interiors)
    text = re.sub(r"^\|.*$", "", text, flags=re.M)

    # Image directives are markup, not prose.
    text = re.sub(r"^!\[.*$", "", text, flags=re.M)
    return len(text.split()), removed


def count(markdown: str) -> dict[str, float]:
    words, removed = countable_words(markdown)
    figures = len(re.findall(r"^\*\*Figure \d+\.", markdown, flags=re.M))
    tables = len(re.findall(r"^\*\*Table [\d.]+ ", markdown, flags=re.M))
    return {
        "words": words, "figures": figures, "tables": tables,
        "publication_units": words / 500 + figures + tables,
        "removed": removed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    parser.add_argument("--cap", type=float, default=25.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    result = count(args.manuscript.read_text(encoding="utf-8"))
    pu = result["publication_units"]
    if args.verbose:
        print("excluded from the word count:")
        for name, n in result["removed"].items():
            print(f"  {n:6d}  {name}")
    print(f"countable words  {result['words']:6d}  -> {result['words']/500:5.1f} PU")
    print(f"figures          {result['figures']:6d}  -> {result['figures']:5.1f} PU")
    print(f"tables           {result['tables']:6d}  -> {result['tables']:5.1f} PU")
    print(f"TOTAL            {pu:11.1f} PU   (AGU threshold {args.cap:.0f})")

    if pu > args.cap:
        over = pu - args.cap
        print(f"OVER by {over:.1f} PU: cut {over * 500:.0f} words, or "
              f"{over:.0f} figure(s)/table(s)")
        return 1
    margin = args.cap - pu
    print(f"inside by {margin:.1f} PU ({margin * 500:.0f} words of headroom)")
    if margin < 1.0:
        print("NOTE: under one unit of margin -- one more figure or table, or "
              "500 more words, crosses the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
