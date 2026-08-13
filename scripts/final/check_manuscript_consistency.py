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


#: Every ledger claim must carry one of these.  The taxonomy exists because a
#: result can be arithmetically reproducible and still not be admissible as a
#: manuscript headline (post-outcome, known-defect upstream, or outcome-
#: conditioned).  See docs/strong_accept_decision_log.md DLOG-027.
CLAIM_STATUSES = (
    "PRIMARY_FROZEN", "DESCRIPTIVE_PROVISIONAL", "POST_OUTCOME_PROMOTED",
    "NOT_USED",
)

#: Spans in which only a PRIMARY_FROZEN claim may appear.  A provisional number
#: may be reported and discussed in its own Results subsection; it may not be a
#: summary-level conclusion of the paper.
HEADLINE_SPANS = ("abstract", "key_point_1", "conclusions")

#: ``POST_OUTCOME_PROMOTED`` is a deliberate third category and not a loophole.
#:
#: The information results of Sections 4.7-4.10 are the largest effects the
#: study measured and are post-outcome, so the two available options were both
#: bad: bury the paper's strongest evidence outside the Abstract, or promote it
#: and let a reader assume it was pre-registered.  This status takes the third
#: option -- promote the number, and require the span that carries it to say
#: what it is.  A span holding a promoted claim must contain one of the phrases
#: below, so deleting the disclosure fails the build rather than silently
#: upgrading a descriptive result to a confirmatory one.
#:
#: The rule is checkable, which is the point.  "We were careful to mention it"
#: is not a control; "the build fails if it is not mentioned" is.
PROMOTION_DISCLOSURES = (
    "post-outcome",
    "descriptive",
)

#: Numbers withdrawn by DLOG-018/025 and never restored.  Matching is
#: boundary-aware so an unrelated 0.48 in another context is not flagged, but
#: any reappearance of the withdrawn headline forms fails the build.
WITHDRAWN_NUMBER_PATTERNS = {
    r"\+\s*0\.48\s*°?C": "withdrawn geometry effect +0.48 degC (DLOG-018)",
    r"(?<![\d.])0\.622\s*°?C": "withdrawn F3 seven-day RMSE 0.622 degC (DLOG-018)",
    r"\b30\s*[x×]\b": "withdrawn 30x information ratio (DLOG-018)",
    r"\b27\s*:\s*1\b": "withdrawn 27:1 information ratio (DLOG-018)",
}

#: Assertions the v4 protocol forbids.  Phrases are chosen so that a *negated*
#: disclaimer ("the as-issued vintage cannot be reconstructed") does not trip
#: the gate while an affirmative claim does.
PROHIBITED_PHRASES = {
    "operational forecast gain": "F3 is a retrospective oracle, not an operational gain",
    "operational forecast improvement": "F3 is a retrospective oracle",
    "deployable gain": "F3 is a retrospective oracle",
    "deployable improvement": "F3 is a retrospective oracle",
    "as-issued forecast value": "no archived-vintage coherence gate has passed",
    "coherent issued forecast": "no archived-vintage coherence gate has passed",
    "operational recovery fraction": "F2a may only recover F3_temperature_only",
    "information budget": "F and L are separate conditional designs, not additive",
    "untouched confirmation": "the 2021-2023 window was already open",
    "preregistered 2021-2023": "the 2021-2023 window was already open",
    "prospectively registered": "no prospective registration exists for this window",
}


def load_ledger() -> list[dict]:
    with open(ROOT / "paper" / "claim_ledger.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_claims() -> pd.DataFrame:
    ledger = load_ledger()
    tables = {path.name: pd.read_parquet(path)
              for path in FINAL.glob("*.parquet")}
    resolved = FR.resolve_claim_ledger(ledger, tables)
    by_id = {claim["claim_id"]: claim for claim in ledger}
    resolved["print_precision"] = resolved["claim_id"].map(
        lambda cid: by_id[cid].get("print_precision"))
    resolved["used_in_mode"] = resolved["claim_id"].map(
        lambda cid: by_id[cid].get("used_in_mode", "all"))
    resolved["claim_status"] = resolved["claim_id"].map(
        lambda cid: by_id[cid].get("status"))
    return resolved


def check_claim_status(ledger: list[dict]) -> list[str]:
    """Every claim declares a status, and provisional claims stay out of headlines.

    This is the gate that makes the Phase-0 quarantine mechanical rather than
    editorial: a number whose experiment is known-defective, post-outcome, or
    conditioned on the realised outcome can still be reported, but it cannot
    reach the Abstract, the Key Points, or the Conclusions.
    """
    problems: list[str] = []
    for claim in ledger:
        cid = claim.get("claim_id", "<unnamed>")
        status = claim.get("status")
        if status not in CLAIM_STATUSES:
            problems.append(
                f"claim {cid} declares status {status!r}; expected one of "
                f"{list(CLAIM_STATUSES)}")
            continue
        used = list(claim.get("used_in") or [])
        if status == "NOT_USED":
            if used:
                problems.append(
                    f"claim {cid} is NOT_USED but still lists used_in={used}")
            continue
        if not used:
            problems.append(f"claim {cid} has status {status} but no used_in spans")
        if status == "DESCRIPTIVE_PROVISIONAL":
            leaked = sorted(set(used) & set(HEADLINE_SPANS))
            if leaked:
                problems.append(
                    f"claim {cid} is DESCRIPTIVE_PROVISIONAL and may not appear "
                    f"in headline spans {leaked}; move it to a Results "
                    f"subsection or promote the evidence first")
    return problems


def check_highlights_match_the_key_points(manuscript: str) -> list[str]:
    """paper/highlights.md declares itself byte-identical to the Key Points.

    It was not: it trailed the manuscript by three revisions, still printing a
    superseded skill value, the pre-attrition cohort size and a Key Point that
    had been replaced. The file said the right thing about itself and nothing
    checked it, which is the failure mode this whole gate exists to catch.

    Comparison is on the bullet text with whitespace collapsed, so a rewrapped
    line is not a difference but a changed word is.
    """
    highlights = ROOT / "paper" / "highlights.md"
    if not highlights.exists():
        return []

    def bullets(text: str, marker: str) -> list[str]:
        block = text.split(marker, 1)
        if len(block) < 2:
            return []
        body = re.split(r"\n## ", block[1], maxsplit=1)[0]
        items = re.findall(r"^\s*(?:-|\d+\.)\s+(.+?)(?=\n\s*(?:-|\d+\.)\s|\n\s*\n|\Z)",
                           body, re.S | re.M)
        return [" ".join(i.split()) for i in items]

    want = bullets(manuscript, "## Key Points")
    got = bullets(highlights.read_text(encoding="utf-8"), "## Key Points")
    if not want:
        return ["cannot locate the manuscript Key Points block"]
    if want == got:
        return []
    problems = [f"paper/highlights.md Key Points differ from the manuscript "
                f"({len(got)} vs {len(want)} items)"]
    for i, (a, b) in enumerate(zip(want, got), 1):
        if a != b:
            problems.append(f"  point {i} manuscript: {a[:88]}")
            problems.append(f"  point {i} highlights: {b[:88]}")
    return problems


def check_promoted_claims_disclose_their_status(
    spans: dict[str, str], ledger: list[dict]) -> list[str]:
    """A promoted post-outcome claim must be labelled where it is promoted.

    Checked against the rendered span rather than against an author's promise,
    because the failure this guards is exactly the one that happens by omission:
    a later edit tightens the Abstract for length, the word "descriptive" goes,
    and a post-outcome result silently becomes a confirmatory headline.
    """
    problems: list[str] = []
    promoted = [c for c in ledger
                if c.get("status") == "POST_OUTCOME_PROMOTED"]
    if not promoted:
        return problems
    used = {span for c in promoted for span in (c.get("used_in") or [])}
    for span in sorted(used & set(HEADLINE_SPANS)):
        text = spans.get(span)
        if not text:
            problems.append(f"cannot locate span {span!r} to check its "
                            "post-outcome disclosure")
            continue
        lowered = text.lower()
        if not any(phrase in lowered for phrase in PROMOTION_DISCLOSURES):
            problems.append(
                f"span {span!r} carries a POST_OUTCOME_PROMOTED claim but "
                f"discloses no status; it must contain one of "
                f"{list(PROMOTION_DISCLOSURES)}")
    return problems


#: Sentences that assert a result is absent from a headline span.  Written as
#: one pattern rather than a list of literals because the real manuscript used
#: the conjoined form -- "outside the Abstract and Key Points" -- which a list
#: of single-span literals matches only half of, and half a match here means
#: the Key Points half of the contradiction goes unreported.
_DENIAL_RE = re.compile(
    r"(?:rather than in|outside|excluded from|not in|withheld from)\s+the\s+"
    r"(abstract|key points)"
    r"(?:\s+(?:and|or|nor)\s+(?:the\s+)?(abstract|key points))?",
    re.IGNORECASE,
)
#: Which ledger span each denial target names.
_DENIAL_SPANS = {"abstract": "abstract", "key points": "key_point_1"}


def check_placement_denials_are_true(
    manuscript: str, spans: dict[str, str], ledger: list[dict]) -> list[str]:
    """A sentence saying "reported outside the Abstract" must be true.

    This is the check that was missing.  Three sections said their results were
    withheld from the Abstract and the Key Points on evidence-grade grounds
    while those very results were the Abstract's core, and every existing gate
    passed: `check_promoted_claims_disclose_their_status` looks for the *word*
    "descriptive", and the word was present in both places.  A manuscript can
    therefore be fully labelled and still lie about where its labels apply.

    The failure mode is ordinary rather than exotic.  The placement sentences
    were written when the results genuinely were confined to Section 4, the
    results were later promoted on request, and nothing connected the promotion
    to the sentences that described the old arrangement.  Prose that describes
    the document's own structure goes stale exactly like a stale comment, and
    it is worth strictly more than a comment because a referee reads it as a
    statement about the authors' discipline.

    So the rule is: if the manuscript denies that a post-outcome result appears
    in a headline span, and a POST_OUTCOME_PROMOTED claim is registered as used
    in that span, the denial is false and the gate fails.  Weakening this to a
    warning would defeat it -- the whole point is that the contradiction is
    invisible to a reader who trusts either half.
    """
    problems: list[str] = []
    promoted_spans = {
        span
        for c in ledger
        if c.get("status") == "POST_OUTCOME_PROMOTED"
        for span in (c.get("used_in") or [])
    }
    for match in _DENIAL_RE.finditer(manuscript):
        named = [g.lower() for g in match.groups() if g]
        for name in named:
            span = _DENIAL_SPANS[name]
            if span in promoted_spans:
                problems.append(
                    f"the manuscript says {match.group(0).strip()!r}, but a "
                    f"POST_OUTCOME_PROMOTED claim is registered as used in "
                    f"{span!r}; the placement sentence contradicts the ledger, "
                    "so one of the two is out of date")
    return problems


#: A withdrawn number or forbidden phrase may still be *narrated* — the
#: decision-log chronology in SI04 has to say what was withdrawn, and the
#: Limitations have to say what F3 is not.  A line carrying one of these
#: markers is therefore reporting the prohibition rather than making the
#: claim.  The vocabulary is deliberately small and explicit: widening it is
#: how a gate like this quietly stops working.
NARRATION_MARKERS = (
    "withdrew", "withdrawn", "superseded", "historical", "retracted",
    "formerly", "initially reported", "no longer", "never", "not an",
    "not a ", "cannot", "must not", "is not", "are not", "rather than",
    "forbidden", "no claim", "does not",
)


def _is_narration(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in NARRATION_MARKERS)


def check_quarantined_content(documents: dict[str, str]) -> list[str]:
    """Withdrawn numbers and protocol-forbidden assertions must not be asserted.

    The scan is line-by-line so that a chronology entry recording a withdrawal
    is distinguishable from a sentence restating the withdrawn number as
    current evidence.
    """
    problems: list[str] = []
    for name, text in documents.items():
        haystack = text.replace("–", "-").replace("−", "-")
        for number, line in enumerate(haystack.splitlines(), start=1):
            if _is_narration(line):
                continue
            for pattern, reason in WITHDRAWN_NUMBER_PATTERNS.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    problems.append(
                        f"{name}:{number} asserts withdrawn value "
                        f"{match.group(0)!r}: {reason}")
            lowered = line.lower()
            for phrase, reason in PROHIBITED_PHRASES.items():
                if phrase in lowered:
                    problems.append(
                        f"{name}:{number} asserts prohibited claim "
                        f"{phrase!r}: {reason}")
    return problems


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
    # Discover the figure numbers rather than hardcoding a range: the range was
    # (1, 2, 3, 4), so inserting a figure left the fifth caption unreadable to
    # the gate and every claim bound to it failed as "not printed" whatever the
    # caption said.
    for n in sorted({int(x) for x in re.findall(r"^\*\*Figure (\d+)\.",
                                                manuscript, re.MULTILINE)}):
        # non-greedy: stop at the FIRST blank line (a greedy `.*` with the
        # `\Z` alternative in the lookahead would swallow the document tail)
        m = re.search(rf"\*\*Figure {n}\..*?(?=\n\n|\Z)", manuscript, re.DOTALL)
        if m:
            out[f"figure_{n}"] = m.group(0)
    # Find the caption by walking back from the generated-block marker rather
    # than by matching a hard-coded "**Table 4.6 ".  Renumbering the tables to a
    # contiguous 1-5 would otherwise have made this `find` return -1 and
    # switched the whole table-of-record check off in silence -- the same
    # failure mode as the exemption hole, reintroduced by a cosmetic edit.
    MARKER = "<!-- TABLE 4.6 (generated) -->"
    gen = manuscript.find(MARKER)
    if gen < 0:
        out["table_4_6"] = ""  # loud: every bound claim now fails as unprinted
        return out
    cap_start = manuscript.rfind("\n\n**Table ", 0, gen)
    if cap_start >= 0:
        start = cap_start + 2
        cap_end = manuscript.find("\n\n", start)
        if cap_end < 0:
            cap_end = len(manuscript)
        cap = manuscript[start:cap_end]
        # the generated block follows the marker after a blank line; rows
        # are contiguous pipe-lines
        m = re.search(r"\n\|[^\n]*\n(?:\|[^\n]*\n)+",
                      manuscript[gen + len(MARKER):])
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


def check_provisional_not_in_headlines(manuscript: str,
                                       resolved: pd.DataFrame) -> list[str]:
    """A provisional value must be absent from the headline spans, not merely undeclared.

    ``check_claim_status`` only reads the ledger's ``used_in`` list, so a
    provisional number could still be printed in the Abstract while the ledger
    claims it lives in Section 4.6.  This check reads the manuscript itself.
    """
    problems: list[str] = []
    spans = {token: text.replace("–", "-").replace("−", "-")
             for token, text in _md_spans(manuscript).items()}
    for row in resolved.itertuples(index=False):
        if row.status != "RESOLVED" or row.claim_status != "DESCRIPTIVE_PROVISIONAL":
            continue
        forms = _printable_forms(
            str(row.value), None, getattr(row, "print_precision", None))
        patterns = _form_patterns(forms)
        for span in HEADLINE_SPANS:
            if _printed(patterns, spans.get(span, "")):
                problems.append(
                    f"claim {row.claim_id} is DESCRIPTIVE_PROVISIONAL but its "
                    f"value {row.value} is printed in {span}; a provisional "
                    f"result may not carry a summary-level statement")
    return problems


def check_generated_tables() -> list[str]:
    problems: list[str] = []
    if not GENERATED.exists():
        return ["generated tables file missing; run generate_manuscript_tables.py"]
    generated = GENERATED.read_text(encoding="utf-8")
    # A generated table has to match the generator wherever it is *published*,
    # and the Supporting Information is published.  Searching only the main text
    # made relocating a table to the SI indistinguishable from deleting it: the
    # hydrologic-state table moved to SI11 on the reviewer's own recommendation
    # and this check called it missing.  The property worth enforcing is that no
    # shipped copy of a generated table drifts from the generator, not that
    # every table lives in the main text.
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    for _si in sorted((ROOT / "paper" / "si").glob("SI*.md")):
        manuscript += "\n" + _si.read_text(encoding="utf-8")
    # every generated table block (contiguous pipe-lines) must appear
    # verbatim in that corpus (whitespace-normalised).  finditer with a
    # non-capturing group is used so the FULL block is matched: with a
    # capturing group, re.findall returns only the group (the block's last
    # line), which would validate a single row per table.
    for match in re.finditer(r"\|.*\n(?:\|.*\n)+", generated):
        block = match.group(0)
        norm = "\n".join(line.strip() for line in block.splitlines())
        norm = norm.replace("\u2013", "-").replace("−", "-")
        if norm and norm not in manuscript.replace("\u2013", "-").replace("−", "-"):
            # This exemption existed for a model whose rows were generated
            # before the manuscript discussed it.  It is keyed on a row *inside*
            # the block, so when an edit removed the last mention of that model
            # the exemption silently switched off the check for the whole table
            # -- which is how a nine-row table of record went missing from the
            # manuscript while this gate reported success.  It now says so.
            if "PlainCausalTCN-7var" in block and "PlainCausalTCN-7var" not in manuscript:
                problems.append(
                    "generated table block is absent from the manuscript and was "
                    "exempted because it carries a PlainCausalTCN-7var row that "
                    "the manuscript no longer mentions; either restore the table "
                    "or retire the exemption:\n" + block[:200])
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
    ledger = load_ledger()
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
    documents = {"manuscript": manuscript}
    if TEX.exists():
        documents["tex"] = TEX.read_text(encoding="utf-8")
    for si_path in sorted((ROOT / "paper" / "si").glob("SI*.md")):
        documents[f"si/{si_path.name}"] = si_path.read_text(encoding="utf-8")
    problems += check_claim_status(ledger)
    problems += check_promoted_claims_disclose_their_status(
        _md_spans(manuscript), ledger)
    problems += check_placement_denials_are_true(
        manuscript, _md_spans(manuscript), ledger)
    problems += check_highlights_match_the_key_points(manuscript)
    problems += check_quarantined_content(documents)
    problems += check_provisional_not_in_headlines(manuscript, resolved)
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
