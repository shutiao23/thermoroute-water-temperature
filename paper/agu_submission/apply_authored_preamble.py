#!/usr/bin/env python3
"""Re-apply the hand-authored preamble to a freshly generated TeX.

This exists for the handover only, and it exists because I needed it three
times in one afternoon. Each regeneration from Markdown reproduced the pandoc
preamble and silently discarded the three preamble decisions the document
depends on -- the build then failed, or produced a 33-page PDF, in a way that
looked unrelated to the regeneration that caused it. That is precisely the
argument in RETIRED_build_agu.md, demonstrated on its author.

After the handover the `.tex` is the manuscript and this script is dead. It is
kept because the three edits are decisions with reasons, and a reader asking
"why is hyperref not loaded" should find the answer in one place rather than in
a commit message.

Idempotent: applying it twice is a no-op, and it reports which edits were
already present so a partially patched file cannot be mistaken for a clean one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "ThermoRoute_WRR.tex"

#: (name, generated form, authored form).  Plain string replacement throughout:
#: an earlier version of this used a regex and lost an afternoon to a ``\c``
#: escape in the replacement text, which is a silly way to break a build.
EDITS: tuple[tuple[str, str, str], ...] = (
    (
        "layout",
        "\\documentclass[draft,linenumbers]{agujournal2025}",
        "% Both options are required for an AGU submission and neither is\n"
        "% cosmetic.  ``linenumbers'' gives referees something to cite;\n"
        "% ``draft'' sets \\draftskip=20 in agujournal2025.cls, which is the\n"
        "% 1.5-2 line spacing AGU asks for in a submitted manuscript.\n"
        "%\n"
        "% This was briefly dropped to reach a 23-page PDF and that was a\n"
        "% mistake: AGU measures length in publication units, not pages.\n"
        "% PU = words/500 + figures + tables, counted over the abstract, body,\n"
        "% captions and appendices only -- references, the plain-language\n"
        "% summary, Open Research, table interiors and the SI are all excluded.\n"
        "% This manuscript is 24.7 PU against a threshold of 25, so it was\n"
        "% already inside the real limit at full double spacing, and the\n"
        "% single-spaced build traded a required format for a metric the\n"
        "% publisher does not use.\n"
        "%\n"
        "% The spacing itself is a choice inside the option, not fixed by it.\n"
        "% AGU asks for \"lines spaced 1.5-2 lines\"; agujournal2025.cls ships\n"
        "% \\draftskip=20, which at the 10pt body is exactly 2.0 and the top of\n"
        "% that range. 15 is 1.5, the bottom of the same range and equally\n"
        "% compliant, and it is worth about five pages here. The class reads\n"
        "% \\draftskip inside \\normalsize, so the counter is set before the\n"
        "% first \\normalsize rather than after \\begin{document}.\n"
        "\\documentclass[draft,linenumbers]{agujournal2025}\n"
        "\\draftskip=15\n"
        "\\makeatletter\\normalsize\\makeatother",
    ),
    (
        "hyperref",
        "\\usepackage{hyperref}",
        "% hyperref is deliberately NOT loaded.  pandoc pulled it in for the\n"
        "% inline DOI links, and those are gone: every citation is a real\n"
        "% apacite key and the document contains no \\href or \\url at all.\n"
        "% Keeping it was not free -- this apacite and this hyperref disagree\n"
        "% about \\hyper@link@, so the build succeeded on a clean tree and then\n"
        "% died on the pass after bibtex, once \\bibcite entries existed for\n"
        "% apacite to hyperlink.  An unused package that breaks the second\n"
        "% build is worse than no package.",
    ),
)

#: The generated bibliography note describes a state that no longer exists.
OLD_BIB_NOTE_MARKER = "TODO(AUTHORS, docs/WRR_SUBMISSION_CHECKLIST.md item 6.4)"
NEW_BIB_NOTE = (
    "%\n"
    "% Conversion done: every citation is a real \\cite or \\citeA key and\n"
    "% \\nocite{*} is gone, so this list is the works the paper actually cites\n"
    "% rather than every entry in ../references.bib.\n"
)


def apply(tex: str) -> tuple[str, list[str], list[str]]:
    applied, already = [], []
    for name, generated, authored in EDITS:
        if authored in tex:
            already.append(name)
        elif generated in tex:
            tex = tex.replace(generated, authored, 1)
            applied.append(name)
        else:
            raise SystemExit(
                f"cannot apply {name!r}: neither the generated nor the authored "
                "form is present; the TeX has diverged from both and needs a "
                "human"
            )

    if OLD_BIB_NOTE_MARKER in tex:
        start = tex.index("%\n% TODO(AUTHORS")
        end = tex.index("\\bibliography{", start)
        tex = tex[:start] + NEW_BIB_NOTE + tex[end:]
        applied.append("bibliography-note")
    else:
        already.append("bibliography-note")
    return tex, applied, already


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=TEX)
    parser.add_argument("--check", action="store_true",
                        help="report without writing; nonzero if edits are missing")
    args = parser.parse_args(argv)

    original = args.tex.read_text(encoding="utf-8")
    patched, applied, already = apply(original)
    print(f"applied: {applied or 'none'}")
    print(f"already present: {already or 'none'}")
    if args.check:
        return 1 if applied else 0
    if patched != original:
        args.tex.write_text(patched, encoding="utf-8")
        print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
