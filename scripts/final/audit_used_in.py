#!/usr/bin/env python3
"""Audit claim ``used_in`` spans against the manuscript prose.

Run BEFORE switching the gate from any->all: reports, per claim, which
declared ``used_in`` spans actually print the value in prose (generated table
blocks stripped).  The output is used to repair ``paper/claim_ledger.yaml`` so
that ``used_in`` lists EXACTLY the spans whose prose must carry the value.

Semantics after the switch: a claim's value must appear in EVERY span listed
in its ``used_in``.  ``used_in_mode: any`` is an explicit escape hatch that
must be accompanied by an ``any_reason`` note in the ledger.

Usage::

    python scripts/final/audit_used_in.py [--show-hits]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "final"))

import check_manuscript_consistency as G


def main() -> int:
    resolved = G.resolve_claims()
    manuscript = G.MANUSCRIPT.read_text(encoding="utf-8")
    si_tokens = {t for row in resolved.itertuples(index=False)
                 for t in str(row.used_in).split(";")
                 if re.fullmatch(r"si\d+", t)}
    spans = G.resolve_si_spans(
        {token: text.replace("\u2013", "-").replace("\u2212", "-")
         for token, text in G._md_spans(manuscript).items()},
        list(si_tokens))
    pv = (G.FINAL / "paper_values.tex").read_text(encoding="utf-8")
    rendered = dict(re.findall(
        r"\\newcommand\{\\([A-Za-z0-9]+)\}\{([^}]*)\}", pv))

    n_fix = 0
    for row in resolved.itertuples(index=False):
        if row.status != "RESOLVED":
            continue
        forms = G._printable_forms(str(row.value),
                                   rendered.get(str(row.latex_macro)),
                                   getattr(row, "print_precision", None))
        pats = G._form_patterns(forms)
        declared = [t for t in str(row.used_in).split(";") if t]
        hit = [t for t in spans if G._printed(pats, spans[t])]
        missing = [t for t in declared if t not in hit]
        extra = [t for t in hit if t not in declared]
        flag = "OK  " if not missing else "FIX "
        if missing:
            n_fix += 1
        print(f"{flag} {row.claim_id:34s} {forms[0]:9s} "
              f"declared={declared}")
        if missing:
            print(f"      missing from prose: {missing}")
        if extra and "--show-hits" in sys.argv:
            print(f"      also printed in:    {extra}")
    print(f"\n{len(resolved)} resolved claims; {n_fix} need ledger repair")
    return 1 if n_fix else 0


if __name__ == "__main__":
    sys.exit(main())
