#!/usr/bin/env python3
"""One-time migration: inline DOI links become real apacite citations.

The Markdown manuscript cited by writing `[Caissie, 2006](https://doi.org/...)`,
which pandoc turned into `\\href{...}{Caissie, 2006}`. Nothing in the document
then carried a citation key, so the bibliography could not be built from what
was cited and the TeX fell back on `\\nocite{*}`: print everything in
`references.bib`, cited or not.

That has three costs, and the page count is the least interesting one. The
reference list stops being a record of what the paper draws on -- 62 entries
printed against 19 actually cited. Author-year citations that a reader can scan
are replaced by raw URLs in the running text. And because `\\nocite{*}` never
fails, deleting the last citation to a work leaves it in the list silently.

This is run once, at the handover to TeX-native authoring. Afterwards the `.tex`
is the manuscript and citations are maintained there by hand, so this script is
provenance rather than part of the build.

Placement follows apacite's two forms, which is why the parenthesis matters:
`(\\href{...}{Caissie, 2006})` is a parenthetical citation and becomes
`\\cite{key}`, while a bare `\\href{...}{Caissie, 2006}` is being used as the
sentence's subject and becomes `\\citeA{key}`, which renders as "Caissie (2006)".
Getting that backwards produces "(Caissie (2006))" and reads as a typo.

A DOI with no entry in `references.bib` is left as a link and reported. Silently
dropping it would delete a citation.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEX = HERE / "ThermoRoute_WRR.tex"
BIB = ROOT / "paper" / "references.bib"

HREF = re.compile(r"\\href\{(?P<url>https?://[^}]+)\}\{(?P<label>[^}]*)\}")


def doi_to_key(bib: str) -> dict[str, str]:
    """Map every DOI and URL in the bibliography to its entry key."""
    mapping: dict[str, str] = {}
    for entry in re.split(r"\n(?=@)", bib):
        key = re.search(r"@\w+\{([^,]+),", entry)
        if not key:
            continue
        for field in ("doi", "url"):
            found = re.search(rf"{field}\s*=\s*[{{\"]([^}}\"]+)", entry, re.I)
            if found:
                doi = (found.group(1).strip().lower()
                       .replace("https://doi.org/", "").rstrip("/"))
                mapping.setdefault(doi, key.group(1))
    return mapping


def migrate(tex: str, mapping: dict[str, str]) -> tuple[str, int, list[str]]:
    unmatched: list[str] = []
    converted = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal converted
        doi = (match.group("url").lower()
               .replace("https://doi.org/", "").rstrip("/"))
        key = mapping.get(doi)
        if key is None:
            unmatched.append(match.group("url"))
            return match.group(0)
        converted += 1
        # The caller strips a wrapping parenthesis pair and passes the form.
        return f"\\citeA{{{key}}}"

    # Parenthetical first, so the parentheses are consumed rather than left
    # around a \citeA that already prints its own.
    def replace_paren(match: re.Match[str]) -> str:
        nonlocal converted
        doi = (match.group("url").lower()
               .replace("https://doi.org/", "").rstrip("/"))
        key = mapping.get(doi)
        if key is None:
            unmatched.append(match.group("url"))
            return match.group(0)
        converted += 1
        return f"\\cite{{{key}}}"

    tex = re.sub(r"\(" + HREF.pattern + r"\)", replace_paren, tex)
    tex = HREF.sub(replace, tex)

    # A citation-driven bibliography must not also print everything else.
    tex = re.sub(r"^\\nocite\{\*\}\n", "", tex, flags=re.M)
    return tex, converted, unmatched


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=TEX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    mapping = doi_to_key(BIB.read_text(encoding="utf-8"))
    original = args.tex.read_text(encoding="utf-8")
    migrated, converted, unmatched = migrate(original, mapping)

    keys = sorted(set(re.findall(r"\\cite[A]?\{([^}]+)\}", migrated)))
    print(f"bibliography DOIs: {len(mapping)}")
    print(f"links converted:   {converted}")
    print(f"distinct works cited: {len(keys)}")
    if unmatched:
        print(f"UNMATCHED (left as links): {len(unmatched)}")
        for url in sorted(set(unmatched)):
            print(f"   {url}")
    if "\\nocite{*}" in migrated:
        print("WARNING: \\nocite{*} survived")
    if not args.dry_run:
        args.tex.write_text(migrated, encoding="utf-8")
        print(f"wrote {args.tex}")
    return 1 if unmatched else 0


if __name__ == "__main__":
    raise SystemExit(main())
