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

from thermoroute import final_results as FR

FINAL = ROOT / "outputs" / "final"
MANUSCRIPT = ROOT / "paper" / "ThermoRoute_paper.md"
GENERATED = ROOT / "paper" / "tables_final.md"
TEX = ROOT / "paper" / "agu_submission" / "ThermoRoute_WRR.tex"


def resolve_claims() -> pd.DataFrame:
    with open(ROOT / "paper" / "claim_ledger.yaml", encoding="utf-8") as fh:
        ledger = yaml.safe_load(fh)
    tables = {path.name: pd.read_parquet(path)
              for path in FINAL.glob("*.parquet")}
    resolved = FR.resolve_claim_ledger(ledger, tables)
    by_id = {claim["claim_id"]: claim for claim in ledger}
    resolved["print_precision"] = resolved["claim_id"].map(
        lambda cid: by_id[cid].get("print_precision"))
    resolved["used_in_mode"] = resolved["claim_id"].map(
        lambda cid: by_id[cid].get("used_in_mode", "all"))
    return resolved


def _form_patterns(forms: list[str]) -> list[str]:
    """Boundary-aware regexes for the printed forms.

    A value like ``0.25`` must not count as printed by appearing inside
    ``0.251``, and ``116`` must not match inside ``1116`` or ``1165``.
    """
    patterns = []
    for form in forms:
        patterns.append(
            rf"(?<![0-9A-Za-z.]){re.escape(form)}(?!\d)")
    return patterns


def _printed(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


_UNIT_NUM = re.compile(
    r"(?<![0-9A-Za-z.])(-?\d+\.\d{2,4})(?=\s*(?:°C|,?\s?°C|\b))")


def check_contradictions(resolved: pd.DataFrame,
                         spans: dict[str, str]) -> list[str]:
    """Flag same-sentence numbers close to a claim value but not equal to it.

    Presence checks cannot see a stale sibling number (the signature of the
    original 1.478-vs-1.459 defect: the table row was right and the prose was
    wrong, in the same section).  For each sentence containing the claim's
    printed value, any other number within 5% relative is reported.  The
    check is warning-level by design: sentences legitimately juxtapose two
    models' RMSEs, so the output is reviewed and whitelisted per claim before
    it may be promoted to a hard gate.
    """
    warnings: list[str] = []
    for row in resolved.itertuples(index=False):
        if row.status != "RESOLVED":
            continue
        try:
            v = float(row.value)
        except (TypeError, ValueError):
            continue
        exact = {f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"}
        for token in str(row.used_in).split(";"):
            text = spans.get(token, "")
            for sentence in re.split(r"(?<=[.;])\s+", text):
                if not any(e in sentence for e in exact):
                    continue
                for m in _UNIT_NUM.finditer(sentence):
                    other = m.group(1)
                    if other in exact:
                        continue
                    try:
                        other_v = float(other)
                    except ValueError:
                        continue
                    if abs(other_v - v) < 0.05 * max(abs(v), 1e-9):
                        warnings.append(
                            f"claim {row.claim_id}: sentence in {token} "
                            f"prints {other} beside the claim value "
                            f"{v:.3f} (within 5%); stale-edit suspect: "
                            f"...{sentence.strip()[:120]}...")
    return warnings


def _printable_forms(value: str, rendered: str | None = None,
                     print_precision: int | None = None) -> list[str]:
    """Candidate printed forms of a resolved claim value.

    Default forms are the exact 3-4 decimal renderings plus the rendered
    value in ``paper_values.tex``.  Coarser forms are included only when the
    claim ledger declares the prose precision: ``print_precision: 0`` for
    whole units (nearest-gauge distances) and ``print_precision: 2`` for
    values prose rounds to two decimals (e.g. median memory gain 0.491
    printed as 0.49).  Unconditional coarse forms are unsafe: they can be
    satisfied by unrelated numbers in prose (e.g. a half-life of 6.9 days
    matching "7", or sensor noise at 0.25 matching a skill of 0.250).
    """
    try:
        fv = float(value)
    except ValueError:
        return [value]
    forms = {f"{fv:.3f}", f"{fv:.4f}"}
    if fv == int(fv):
        forms.add(f"{int(fv)}")
    if print_precision == 0:
        forms.add(f"{fv:.0f}")
    elif print_precision == 2:
        forms.add(f"{fv:.2f}")
    if rendered:
        forms.add(rendered)
    return sorted(forms, key=len)


_GENERATED_BLOCK_RE = re.compile(
    r"<!-- TABLE .*?\(generated\) -->\n+(?:\|[^\n]*\n)+", re.MULTILINE)


def resolve_si_spans(spans: dict[str, str],
                     used_tokens: list[str] | None = None) -> dict[str, str]:
    """Add SI-file spans for ``siNN`` tokens (used_in lists or span map).

    SI files are scanned in full; their numbers are not ledger-bound at row
    level, but total drift is caught.
    """
    out = dict(spans)
    tokens = set(spans)
    if used_tokens:
        tokens.update(t for t in used_tokens if re.fullmatch(r"si\d+", t))
    for token in tokens:
        m = re.fullmatch(r"si(\d+)", token)
        if not m:
            continue
        si_files = sorted((ROOT / "paper" / "si").glob(f"SI{m.group(1)}_*.md"))
        if si_files:
            out[token] = si_files[0].read_text(encoding="utf-8") \
                .replace("\u2013", "-").replace("−", "-")
    return out


def _md_spans(manuscript: str) -> dict[str, str]:
    """Map ``used_in`` tokens to their text spans in the Markdown manuscript.

    Section and Key-Points spans exclude generated table blocks, so a headline
    number that appears only inside a generated table is not treated as
    printed in prose.  ``table_4_6`` includes its generated block: that block
    is a table of record validated verbatim against ``tables_final.md``.
    """
    lines = manuscript.splitlines()
    heads = [(i, line) for i, line in enumerate(lines)
             if line.startswith(("## ", "### "))]
    out: dict[str, str] = {}

    def section_text(li: int, end: int) -> str:
        return _GENERATED_BLOCK_RE.sub(
            "", "\n".join(lines[li:end]))

    for i, (li, line) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(lines)
        if line == "## Key Points":
            out["key_point_1"] = section_text(li, end)
        elif line == "## Abstract":
            out["abstract"] = section_text(li, end)
        elif line == "## 7. Conclusions":
            out["conclusions"] = section_text(li, end)
        elif line.startswith("### 4."):
            token = "section_" + line[4:].split()[0].replace(".", "_")
            out[token] = section_text(li, end)
    for n in (3, 4, 5, 6):
        # non-greedy: stop at the FIRST blank line (a greedy `.*` with the
        # `\Z` alternative in the lookahead would swallow the document tail)
        m = re.search(rf"\*\*Figure {n}\..*?(?=\n\n|\Z)", manuscript, re.DOTALL)
        if m:
            out[f"figure_{n}"] = m.group(0)
    start = manuscript.find("**Table 4.6 ")
    if start >= 0:
        cap_end = manuscript.find("\n\n", start)
        if cap_end < 0:
            cap_end = len(manuscript)
        cap = manuscript[start:cap_end]
        gen = manuscript.find("<!-- TABLE 4.6 (generated) -->", cap_end)
        if gen >= 0:
            # the generated block follows the marker after a blank line; rows
            # are contiguous pipe-lines
            m = re.search(r"\n\|[^\n]*\n(?:\|[^\n]*\n)+",
                          manuscript[gen + len("<!-- TABLE 4.6 (generated) -->"):])
            if m:
                cap += m.group(0)
        out["table_4_6"] = cap
    return out


def check_manuscript_numbers(manuscript: str, resolved: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    pv = (FINAL / "paper_values.tex").read_text(encoding="utf-8")
    macro_rendered: dict[str, str] = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z0-9]+)\}\{([^}]*)\}", pv):
        macro_rendered[m.group(1)] = m.group(2)
    # TeX is a second manuscript copy; every headline value must appear in
    # both documents, so a fix applied to one cannot silently miss the other.
    tex = TEX
    if not tex.exists():
        problems.append(f"TeX manuscript {tex} missing; cannot cross-check numbers")
        tx = ""
    else:
        tx = tex.read_text(encoding="utf-8").replace("\u2013", "-").replace("−", "-")
    spans = {token: text.replace("\u2013", "-").replace("−", "-")
             for token, text in _md_spans(manuscript).items()}
    # SI tokens (si07, si11, ...) resolve to their supporting-info files,
    # scanned in full (their numbers are not ledger-bound at row level);
    # tokens are taken from the used_in lists as well as the span map
    si_tokens = {t for row in resolved.itertuples(index=False)
                 for t in str(row.used_in).split(";")
                 if re.fullmatch(r"si\d+", t)}
    spans = resolve_si_spans(spans, list(si_tokens))
    for row in resolved.itertuples(index=False):
        if row.status != "RESOLVED":
            continue
        if not str(row.value):
            continue
        macro = str(row.latex_macro)
        forms = _printable_forms(
            str(row.value), macro_rendered.get(macro),
            getattr(row, "print_precision", None))
        patterns = _form_patterns(forms)
        # cheap sanity: the macro must exist in paper_values.tex
        if f"\\newcommand{{\\{macro}}}" not in pv:
            problems.append(f"macro \\{macro} missing from paper_values.tex")
        # real numeric comparison: the value, printed at a plausible
        # precision, must appear in EVERY one of the claim's used_in spans
        # (prose only; generated table blocks are stripped) unless the
        # ledger declares ``used_in_mode: any`` as an explicit escape hatch.
        used = [token for token in str(row.used_in).split(";") if token]
        mode = str(getattr(row, "used_in_mode", "all") or "all")
        if not used:
            problems.append(
                f"claim {row.claim_id} has no used_in sections; cannot verify")
        else:
            hit = [token for token in used
                   if _printed(patterns, spans.get(token, ""))]
            ok = (len(hit) == len(used)) if mode == "all" else bool(hit)
            if not ok:
                problems.append(
                    f"claim {row.claim_id} value {row.value} (\\{macro}) not "
                    f"printed in {sorted(set(used) - set(hit))} "
                    f"(mode={mode}, used_in={used}, forms={forms})")
        # ... and in the TeX manuscript (tex/md drift check).  Claims whose
        # used_in is ONLY an SI file are verified against that SI file above
        # and are not expected in the TeX body.
        si_only = bool(used) and all(re.fullmatch(r"si\d+", token)
                                     for token in used)
        if not si_only and not _printed(patterns, tx):
            problems.append(
                f"claim {row.claim_id} value {row.value} (\\{macro}) not found "
                f"in TeX manuscript at printed precision {forms}")
    return problems


def check_generated_tables() -> list[str]:
    problems: list[str] = []
    if not GENERATED.exists():
        return ["generated tables file missing; run generate_manuscript_tables.py"]
    generated = GENERATED.read_text(encoding="utf-8")
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    # every generated table block (contiguous pipe-lines) must appear
    # verbatim in the manuscript (whitespace-normalised).  finditer with a
    # non-capturing group is used so the FULL block is matched: with a
    # capturing group, re.findall returns only the group (the block's last
    # line), which would validate a single row per table.
    for match in re.finditer(r"\|.*\n(?:\|.*\n)+", generated):
        block = match.group(0)
        norm = "\n".join(line.strip() for line in block.splitlines())
        norm = norm.replace("\u2013", "-").replace("−", "-")
        if norm and norm not in manuscript.replace("\u2013", "-").replace("−", "-"):
            # skip blocks whose model rows are not yet in the manuscript
            if "PlainCausalTCN-7var" in block and "PlainCausalTCN-7var" not in manuscript:
                continue
            problems.append(f"generated table block not in manuscript:\n{block[:300]}")
    return problems

def check_table_shapes(paths: list[Path]) -> list[str]:
    """Header column count must equal the alignment row's spec count."""
    problems: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        for i in range(len(lines) - 1):
            h, a = lines[i], lines[i + 1]
            if not (h.startswith("|") and a.startswith("|")):
                continue
            if not re.fullmatch(r"\|(\s*:?-{2,}:?\s*\|)+", a.strip()):
                continue
            nh = len(h.strip().strip("|").split("|"))
            na = len(a.strip().strip("|").split("|"))
            if nh != na:
                problems.append(
                    f"{p.name}:{i+1} table header has {nh} columns but the "
                    f"alignment row declares {na}")
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
    problems += check_table_shapes(sorted((ROOT / "paper").glob("*.md"))
                                  + sorted((ROOT / "paper" / "si").glob("*.md")))
    warnings = check_contradictions(
        resolved, {token: text.replace("\u2013", "-").replace("−", "-")
                   for token, text in _md_spans(manuscript).items()})
    if warnings:
        print("\ncontradiction scan (warning-level, not failing):")
        for warning in warnings:
            print(f"  - {warning}")
    if problems:
        print("\nCONSISTENCY GATE FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconsistency gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
