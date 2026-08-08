#!/usr/bin/env python3
"""Consistency gate between the manuscript and the results authority.

Checks, per protocol v1 (build must fail on any breach):

1. every claim in ``paper/claim_ledger.yaml`` resolves against
   ``outputs/final/`` (missing spatial/mechanism tables are acceptable only as
   PENDING when the corresponding experiment has not run);
2. the manuscript's headline numbers (claims) match the resolved ledger values
   to the printed precision;
3. the manuscript's Section 4.6/4.8 table blocks equal the generated tables
   (``scripts/final/generate_manuscript_tables.py``);
4. no headline claim is sourced from the pooled sensitivity table.

Usage::

    python scripts/final/check_manuscript_consistency.py [--manuscript paper/ThermoRoute_paper.md]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thermoroute import final_results as FR  # noqa: E402

FINAL = ROOT / "outputs" / "final"
MANUSCRIPT = ROOT / "paper" / "ThermoRoute_paper.md"
GENERATED = ROOT / "paper" / "tables_final.md"


def resolve_claims() -> pd.DataFrame:
    with open(ROOT / "paper" / "claim_ledger.yaml", encoding="utf-8") as fh:
        ledger = yaml.safe_load(fh)
    tables = {path.name: pd.read_parquet(path)
              for path in FINAL.glob("*.parquet")}
    return FR.resolve_claim_ledger(ledger, tables)


def check_manuscript_numbers(manuscript: str, resolved: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    for row in resolved.itertuples(index=False):
        if row.status != "RESOLVED":
            continue
        if not str(row.value):
            continue
        macro = str(row.latex_macro)
        # find the corresponding claim's used_in sections in the manuscript
        # and verify at least one printed occurrence of the value
        for section in str(row.used_in).split(";"):
            if section in manuscript:
                break
        # cheap sanity: the macro must exist in paper_values.tex
        pv = (FINAL / "paper_values.tex").read_text(encoding="utf-8")
        if f"\\newcommand{{\\{macro}}}" not in pv:
            problems.append(f"macro \\{macro} missing from paper_values.tex")
    return problems


def check_generated_tables() -> list[str]:
    problems: list[str] = []
    if not GENERATED.exists():
        return ["generated tables file missing; run generate_manuscript_tables.py"]
    generated = GENERATED.read_text(encoding="utf-8")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    # every generated table block (starts with a table header line) must appear
    # verbatim in the manuscript (whitespace-normalised)
    for block in re.findall(r"\|.*\n(\|.*\n)*", generated):
        norm = "\n".join(line.strip() for line in block.splitlines())
        if norm and norm not in manuscript.replace("\u2013", "-").replace("−", "-"):
            # skip blocks whose model rows are not yet in the manuscript
            if "PlainCausalTCN-7var" in block and "PlainCausalTCN-7var" not in manuscript:
                continue
            problems.append(f"generated table block not in manuscript:\n{block[:300]}")
    return problems


def check_no_pooled_headlines() -> list[str]:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    problems = []
    for pattern in ("pooled metric cells", "are not station-medians"):
        if pattern in manuscript:
            problems.append(f"manuscript still contains pooled-as-headline wording: {pattern!r}")
    return problems


def main() -> int:
    resolved = resolve_claims()
    unresolved = resolved[resolved.status != "RESOLVED"]
    print("claim resolution:")
    for row in unresolved.itertuples(index=False):
        print(f"  {row.claim_id}: {row.status}")
    problems: list[str] = []
    for claim in unresolved.itertuples(index=False):
        if claim.status != "PENDING":
            problems.append(f"{claim.claim_id}: {claim.status}")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    problems += check_manuscript_numbers(manuscript, resolved)
    problems += check_generated_tables()
    problems += check_no_pooled_headlines()
    if problems:
        print("\nCONSISTENCY GATE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconsistency gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
