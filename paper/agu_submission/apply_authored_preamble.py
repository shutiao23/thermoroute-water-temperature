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
        "% ``linenumbers'' without ``draft''.  The two are separable and only\n"
        "% one is a review requirement: linenumbers gives referees something to\n"
        "% cite, draft additionally double-spaces the body.  Double-spaced this\n"
        "% manuscript is 33 pages; single-spaced with line numbers it is inside\n"
        "% the 23-page cap, and no content differs between them.  Add ``draft''\n"
        "% back if the editor requires double spacing.\n"
        "\\documentclass[linenumbers]{agujournal2025}",
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
