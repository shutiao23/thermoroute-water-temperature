# Submission-package polish — 2026-08-05

| Field | Value |
|---|---|
| Branch | `feat/route-a-completion` |
| Scope | `paper/**` and `docs/**` only |
| Deliberately not touched | `src/`, `scripts/`, `tests/`, `protocols/`, `pyproject.toml`, `requirements*.txt`, `.github/`, `outputs/`, `ops/`, and both sibling worktrees (`…-multicore`, `…-remediation`, read-only) |
| Commands run | none under `scripts/`; no pytest; no git commit |
| Predecessor | `docs/PAPER_RESTRUCTURE_20260805.md` (main-body restructure, same day) |
| Successor | `docs/WRR_SUBMISSION_CHECKLIST.md` (what remains, by blocker category) |

This document records what changed in each file, every placeholder left behind
and what must fill it, every reference flagged, and every inconsistency found
between the manuscript, the SI, and the figure manifest.

**No number was invented.** Every quantity that appears in a file changed here
was carried forward unchanged from the manuscript restructure, or read from a
named file in this repository. The one-time 2021–2023 evaluation has not
happened, and no file changed here asserts an evaluation-period result.

---

## 1. Files changed

| File | Change |
|---|---|
| `paper/ThermoRoute_paper.md` | Author block, Key Points, citations, Open Research, Acknowledgments, Supporting Information (§2) |
| `paper/highlights.md` | Rewritten as the AGU Key Points source (§3) |
| `paper/cover_letter.md` | Rewritten to the new framing (§4) |
| `paper/agu_submission/build_agu.py` | Generator brought into correspondence with the rewritten Markdown (§5) |
| `paper/agu_submission/README.md` | Rewritten; records the two build blockers and the stale TeX (§5) |
| `paper/references.bib` | Reconciled one-to-one with in-text citations (§6) |
| `paper/si/SI00_inventory.md` | Figure inventory added; SI08 row corrected; POST checks extended (§7) |
| `paper/si/SI08_probability_metrics_RECEIPT.md` | Corrected for the Stage-19 disposition (§7) |
| `paper/si/README.md` | SI08 role, FigS1–FigS9, wording ban (§7) |
| `paper/FIGURE_REDRAW_SPEC.md` | Header said "eight SI figures"; its own matrix lists nine (§8.1) |
| `docs/WRR_SUBMISSION_CHECKLIST.md` | **New.** Every submission item with a status and the single input that closes it |
| `docs/PAPER_POLISH_20260805.md` | **New.** This file |

`paper/agu_submission/ThermoRoute_WRR.tex` was **not** edited. It is a build
artifact; see §5.

---

## 2. `paper/ThermoRoute_paper.md`

### 2.1 Author block

The previous block asserted a three-author, two-affiliation structure
(`[Author One]`, `[Author Two]`, `[Author Three]`). That is an invented shape:
neither the number of authors nor the number of affiliations is known. Replaced
by an explicit `[AUTHOR LIST TO BE COMPLETED]` instruction naming the field set
required per author, two clearly labelled affiliation placeholders with a note to
add or delete lines, and a separate corresponding-author placeholder. The block
points at the signed intake schema in
`docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2.

### 2.2 Key Points

**The previous three Key Points violated AGU's length limit** — AGU permits at
most three Key Points of at most 140 characters each, and the three were 201,
157, and 201 characters (measured on the superseded bytes, whitespace
normalised). All three were rewritten to 134, 130, and 135
characters, each one complete sentence, each stating a checkable fact rather than
an evaluative claim. The new set is mirrored byte-for-byte in
`paper/highlights.md`, and `build_agu.py` now reads it from the Markdown and
rejects a build if the count, the length, or the sentence form is wrong.

### 2.3 Citations added

Fourteen citations were added to statements the manuscript already made, in order
to reconcile `paper/references.bib` (§6). No statement was changed to accommodate
a citation, and no citation carries a DOI or URL that was not already recorded in
the bibliography or already present in the manuscript.

| Location | Citation added | Statement it supports |
|---|---|---|
| §3.2 | Kingma and Ba, 2015; Loshchilov and Hutter, 2019 | Neural members are fitted by adaptive stochastic gradient descent with decoupled weight decay. Verified against `src/thermoroute/train.py:500` (`torch.optim.AdamW`) and `src/thermoroute/development_controls_gate.py:732` — read-only. |
| §3.4 | Rahmani et al., 2021b | Ungauged prediction is a different problem and a different evaluation |
| §3.5 | Beven and Binley, 1992; Kavetski et al., 2006a, 2006b | The Bayesian/likelihood tradition that the distribution-free calibration step is chosen against |
| §3.5 | Künsch, 1989 | Block, rather than row, resampling for a dependent series (the block-maximum calibration variant) |
| §3.5 | Gneiting and Raftery, 2007 | The three-quantile pinball summary and CRPS are distinct proper scoring rules |
| §3.6 | Nash and Sutcliffe, 1970; Gupta et al., 2009 | Why RMSE and paired RMSE differences are reported rather than an efficiency-type criterion |
| §3.6 | Diebold and Mariano, 1995; Harvey et al., 1997 | The standard equal-predictive-accuracy tests, and why they are not used |
| §3.6 | Richardson, 2000 | Relative economic value in the cost–loss sense |
| §3.7 | Hodson and Hariharan, 2023 | The USGS `dataretrieval` Python client used by the acquisition path. Verified against `src/thermoroute/usgs.py:84,397` and `pyproject.toml:16` — read-only. |
| §5.3 | Luo et al., 2025 | Architectures that encode geographic context across regions and scales |
| §6.3 | Wilks, 2011 | The standard event-probability verification metrics that are absent |
| §8 | Virtanen et al., 2020; Pedregosa et al., 2011; Ke et al., 2017; Paszke et al., 2019; Hodson and Hariharan, 2023 | The software stack. Verified against `pyproject.toml:14–16` — read-only. |

`Rahmani et al., 2021` was disambiguated to `2021a` at both existing occurrences,
because a second 2021 Rahmani entry is now cited.

The Wilks (2011) citation is an unlinked parenthetical matching its bibliography
entry, which carries an ISBN and no verified URL. Four other citations — Kingma
and Ba, Loshchilov and Hutter, Pedregosa et al., Paszke et al. — are likewise
unlinked because their entries carry no DOI or recorded URL.

### 2.4 Open Research (§8) — rewritten

The previous §8 contained two statements that **contradict the repository's own
rights record** and were corrected:

1. It said "Primary observations are redistributed from public providers." Per
   `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3, USGS/NWIS response bytes, Daymet
   V4 subsets and any derived field materially encoding Daymet values, gridMET
   responses and derived fields, and mixed derived panels and registries all
   default to **`UNRESOLVED_EXCLUDE`**. Nothing authorizes redistribution today.
2. It stated the derived panel is archived at a DOI, without recording that the
   derived panel is itself in the unresolved class and that
   `docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §1 marks the public archive
   as correctly fail-closed.

The rewritten §8 keeps the AGU-required structure and states the constraints:

- **Data availability** — the deposit is described as *planned and specified*,
  not as existing, with the reason (byte-level rights review incomplete; a
  derived product does not inherit the most permissive upstream terms).
- **Primary observational sources** — USGS NWIS, Daymet V4 R1, gridMET, each with
  its own DOI, each stated to retain its provider's terms independently of this
  manuscript.
- **Material that cannot currently be redistributed** — new. Specifies the
  fallback: product identifier and version, exact request specification,
  retrieval code, and a SHA-256 manifest of the bytes as retrieved, so a third
  party can re-acquire and verify without relying on redistribution. Names the
  four unresolved classes, and names two classes excluded outright — the AGU
  LaTeX class (`EXCLUDE_PUBLIC` in the matrix) and the three legacy CSV files.
- **Software availability** — MIT (SPDX `MIT`), read from the repository
  `LICENSE` and `pyproject.toml:11`, with the explicit scope limit that MIT
  covers the source code only and licenses neither the data, the archived
  provider responses, the typesetting class, nor any dependency binary.
- **Reproduction** — now states plainly that independent clean-host reproduction
  has *not* been carried out and that the statement is not a claim that it has.
- **Placeholder inventory** — five placeholders, all in the
  `[… TO BE MINTED / ASSIGNED / CONFIRMED]` form, with an explicit prohibition on
  substituting a reserved, draft, or example DOI.

The previous text's `[DOI: 10.5281/zenodo.XXXXXXX]` and
`[DOI: 10.5281/zenodo.YYYYYYY]` forms were removed: they read as real Zenodo
identifiers with the digits masked, which is exactly the mistake a placeholder
should not invite.

The `### `-level subheadings first written into §8 were flattened to bold run-in
labels, because AGU numbers sections itself and a numbered `\subsection` under an
unnumbered `\section*{Open Research}` would inherit the previous section's
number.

### 2.5 Acknowledgments — rewritten

Now carries four separately marked placeholders — funding with award numbers,
computational resources, competing interests, and CRediT roles enumerating all
fourteen terms — plus a real, non-placeholder data-provider acknowledgment for
USGS, ORNL DAAC, and the Northwest Knowledge Network, with the explicit statement
that acknowledgment is not endorsement.

### 2.6 Supporting Information — corrected

The paragraph said "Supporting figures S1–S8". The current manifest has **nine**
supporting figures. Rewritten to describe S1–S3 (pre-opening, no
evaluation-period quantity), S4–S8 (evaluation-period expansions of §4.6), and S9
(optional, development-period, never to be compared numerically with an
evaluation-period figure).

### 2.7 Invariants re-verified after every edit

| Invariant | Result |
|---|---|
| `[TO BE FILLED AFTER OPENING]` markers | 15 (unchanged) |
| `ROUTE_A_CLAIM` blocks | 8, joined length 5,982 bytes, SHA-256 `63698d1e52e58d893ba971df54ca7d1f2c9fc04e4151083b73c876b251946cee` — byte-identical to the value recorded in `PAPER_RESTRUCTURE_20260805.md` §5.1 |
| Allowlisted legacy three-site sentences | both present, each twice, verbatim under whitespace normalisation |
| Registry `free_text_lints` (21 regexes) | 3 hits, all inside preserved claim blocks; zero new hits |
| B-02 forbidden verbs | every occurrence prohibitive, in-block, or a neutral non-claim use |
| Phrase "quantile crossing" | absent |

---

## 3. `paper/highlights.md`

Rewritten from a 22-bullet mixed design/status list — which described the
superseded structure, named stage numbers and receipt matrices, and carried a
"Current status" engineering to-do list — into the AGU Key Points source.

It now carries: the AGU constraint stated explicitly; the three Key Points
byte-identical to the manuscript with their character counts; a table giving, for
each Key Point, the checkable assertion and the manuscript section and SI file
where it is specified; the standing scope statements; and a note that its bytes
are hash-frozen in the claim registry and that this revision invalidates that
binding.

One wording change for safety: a table column originally headed "Where it is
established" was renamed "Where it is specified", because *establish* is on the
B-02 forbidden-verb list and the column sat beside Route-A Key Point rows.

---

## 4. `paper/cover_letter.md`

Rewritten. The previous draft opened "not ready for submission", advertised the
superseded title, described Stage-09 matrices and erratum mechanics, and listed
outstanding engineering work.

The new letter is written in its final form and is organised around the five
design decisions the manuscript now argues: the strong baseline suite, mechanical
leakage control, the honestly labelled gauged-transfer arm, conformal intervals
with their assumptions stated, and the single acquisition of the evaluation
labels. It devotes a full paragraph to the pre-specified inference gate, states
that the gate fails on cohort geometry alone, and says the failure is reported
rather than hidden.

**It does not promise a superiority result.** It states explicitly that the
manuscript makes no claim of superiority, non-inferiority, equivalence, or parity
over any reference model and no national generalisation, and that where the
development evidence indicates a well-tuned tree ensemble attains lower
station-median error than the proposed architecture, the manuscript says so.

A blockquoted **send condition** at the head states that the letter must not be
sent until the one-time evaluation has been executed, §4.6 is filled, and every
placeholder is closed.

`achieved coverage` was changed to `coverage` in the conformal paragraph, because
*achieve* is on the B-02 forbidden-verb list; the manuscript's own longstanding
"achieved empirical marginal coverage" usage in §3.5 and Table 4.4 was left
alone, as it predates this revision and is a neutral technical term there.

---

## 5. `paper/agu_submission/` — generator fixed, output not hand-edited

### 5.1 Generator change rationale

`ThermoRoute_WRR.tex` **is generated from the Markdown** by `build_agu.py`, whose
`--check` mode fails when the checked-in bytes are not exactly what the generator
would produce. Hand-editing the output would therefore desync it permanently.
The generator was fixed and the output left untouched.

### 5.2 What was wrong with the generator

It could not have produced a correct TeX from the rewritten Markdown even if it
had run:

| Defect | Fix |
|---|---|
| `REQUIRED_STATUS_TEXT` demanded the phrases "no current performance result" and "no empirical performance conclusion", both deleted by the restructure. The generator would have raised on every build. | Replaced with two phrases present in the new "Manuscript status" block |
| `KEYPOINTS` was a hardcoded tuple describing the superseded structure, and would have contradicted the Markdown's own Key Points block | Removed. Key Points are now extracted from the Markdown, and the extractor enforces exactly three items, ≤ 140 characters each, each ending in a full stop |
| The abstract regex ran to `## 1.`, swallowing the Plain Language Summary and the keyword list into `\begin{abstract}` — visible in the stale TeX, where "Keywords:" sits inside the abstract | Three separate extractions: abstract, Plain Language Summary, keywords |
| No Plain Language Summary output at all, though AGU requires one | Emitted as `\section*{Plain Language Summary}` immediately after the abstract. `agujournal2019.cls` predates the dedicated environment — verified, the class defines `\keypoints` but no PLS environment — so an unnumbered section is the correct construct |
| AGU back matter would have been emitted as ordinary numbered sections | `_agu_back_matter()` routes `Open Research` and `Supporting Information` to `\section*` and `Acknowledgments` to the class's `\acknowledgments`, and raises if there is not exactly one Acknowledgments section |
| Author front matter was a single-affiliation placeholder inconsistent with the manuscript | Two labelled affiliation placeholders plus a comment block naming the intake schema |
| Nothing checked that the result-slot markers survived conversion | `RESULT_SLOT_MARKER_COUNT = 15` is checked in the Markdown *and* re-checked in the converted body; a conversion that lost a marker would present an empty cell as a result |
| Nothing enforced the Stage-19 wording ban | `BANNED_PHRASES` refuses "quantile crossing" |

Extraction and validation were tested against the current Markdown without
pandoc: title, three Key Points at 134/130/135 characters, a 2,737-character
abstract terminating correctly before the Plain Language Summary, a
960-character PLS, the keyword list, and an 88,875-character body carrying all
fifteen markers.

### 5.3 Why the TeX could not be regenerated

> **Correction, 2026-08-05 (AGU-2025 migration).** Blocker 1 below was wrong.
> `pandoc` **is** available: pypandoc vendors it at
> `…/envs/route-a/lib/python3.12/site-packages/pypandoc/files/pandoc`, off `PATH`
> but exactly where `_pandoc_path()` already looks. The original search used the
> conda *base* interpreter, which lacks pypandoc, and concluded absence. Use
> `/home/lzq/anaconda3/envs/route-a/bin/python` for every invocation. Blocker 2
> was and remains correct. See `docs/AGU2025_TEMPLATE_MIGRATION.md`, where the
> TeX is regenerated and compiled to a 44-page PDF.

Two independent blockers, both outside `paper/`:

1. ~~**`pandoc` is absent.** Neither `pandoc` nor `pypandoc` is installed, and a
   filesystem search found no binary. `build_agu.py` requires one.~~
   **Withdrawn — see the correction above; pandoc was present all along.**
2. **The PRE-OPEN render guard refuses, correctly.**
   `assert_preopen_manuscript_render_allowed` requires
   `paper/ThermoRoute_paper.md`, `paper/highlights.md`, and
   `paper/cover_letter.md` to match `preopen_document_sha256` in
   `protocols/route_a_claim_registry_v1.json`. The manuscript already failed
   before this revision; highlights and the cover letter now fail too. Re-sealing
   is a separately authorized `protocols/` change.

`ThermoRoute_WRR.tex` therefore still carries the superseded structure — old
title, old abstract, sections "Problem and scope" through "Conclusion". Its own
frozen hash still *matches* (`34a34af4…`), which is the dangerous case: the file
is seal-valid and content-stale simultaneously. `paper/agu_submission/README.md`
now states this in a "Current build blockers" section, and checklist item 8.4
requires it to be re-sealed only *after* regeneration, together with items
8.1–8.3, so that all four documents enter one lineage.

### 5.4 Reference list

The generated TeX contains no `\cite` and no `\bibliography`, so it renders no
reference list — a submission blocker that predates this revision. The Markdown cites
in linked author-year prose. Converting the convention to `\cite`/`\citeA` keys
against `../references.bib` (class default style `apacite`; the class does define
`\citeA`) is an authorial decision about citation handling, so it is recorded as
checklist item 6.4 rather than imposed here. A LaTeX comment block explaining
this is emitted into the generated TeX so the omission cannot be mistaken for a
bug.

---

## 6. `paper/references.bib`

Reconciled in both directions against the rewritten manuscript.

- **Every in-text citation resolves to an entry.** Verified for all 22 original
  citations and all 14 added ones.
- **Every entry is now cited**, after the 14 additions of §2.3 and one removal.
- **Removed:** `decicco2024dataretrieval` — the *R* `dataRetrieval` package. It
  was uncited and is not what the code uses: `src/thermoroute/usgs.py:84,397`
  imports the Python `dataretrieval` client, whose entry is
  `hodson2023dataretrieval`. Recoverable at commit `b0699a8`.
- **Corrected:** `usgs2024nwis`'s `howpublished` said "Web services accessed via
  dataRetrieval", naming the R package. Changed to name the Python client and to
  record statistic code `00003` alongside the parameter codes, matching §3.7.
- **Added `url` fields** to four DOI-less entries — `romano2019conformalized`,
  `martins2016softmax`, `shazeer2017outrageously`, `ke2017lightgbm` — using the
  exact links the manuscript already carried. No new identifier was created.
- **Header updated** from 42 to 41 entries, with a dated reconciliation note.
- **Structural check:** no duplicate keys, balanced braces in all 41 entries.

### 6.1 References flagged

| Entry | Flag |
|---|---|
| `kingma2015adam`, `loshchilov2019decoupled` | Cited in §3.2, but as unlinked parentheticals: the entries carry no DOI and no recorded URL. Hyperlinking requires verified arXiv or OpenReview identifiers in the `.bib`. |
| `pedregosa2011scikit`, `paszke2019pytorch` | Same situation, cited in §8 |
| `wilks2011statistical` | Same situation, cited in §6.3; the entry carries an ISBN and no verified URL. |
| `romano2019conformalized`, `martins2016softmax`, `shazeer2017outrageously`, `ke2017lightgbm` | No DOI exists in the entry; the `url` now recorded is the one the manuscript already used. A DOI should be added if one is found. |
| All 41 | The header claims every entry was web-verified against Crossref/DataCite/arXiv/DBLP per `outputs/reports/reference_audit_v2.md`. **That claim was not re-verified in this audit** — no network access was used, and `outputs/` was outside the audit scope. Treat it as inherited, not re-attested. |

**No TODO entry was needed.** Every statement in the manuscript that requires a
citation has one, and no citation was needed for which no verified entry existed.

---

## 7. Supporting Information

### 7.1 `SI08_probability_metrics_RECEIPT.md` — the substantive correction

SI08's content was contradicted by today's Stage-19 disposition. The file
presented a single probability/reliability projection with no indication that its
two candidate producers compute overlapping metric families **on different
periods by different code paths**:

- the development-period probabilistic stage, which is **not produced** for this
  submission; and
- the trusted scorer inside the one-time opening, which is the **target-period**
  authority and is unaffected.

A reader of the old SI08 could reasonably have concluded either that the whole
file was void or that development-period probability metrics would appear in it.
Both are wrong. Corrections applied:

1. A scope-correction note stating that §1 and §2 are target-period projections
   only, and that no development-period probability value will ever appear.
2. Section headings added to the two existing tables so their period is explicit.
3. A new §3 carrying the disposition: the §2.1 measured table **verbatim** as
   the disposition document requires — 26,993,675 member-level rows, **0** strict
   ordering violations, **135** zero-width rows (0.0005%), maximum monotonicity
   violation exactly 0.000 °C, 12 distinct sites, h=1 129 / h=3 6 / h=7 0,
   LightGBM 123 / per-station 12, 95.6% with observed target below 1 °C.
4. The wording ban stated in the file itself: "quantile crossing" must not appear;
   zero strict violations were measured; the correct term is "zero-width
   (degenerate) nominal interval".
5. Why the contract was not amended, that no delivered interval is degenerate,
   what is therefore absent from the submission, and the pre-opening
   member-averaged degeneracy check that must be run before the one-shot opening.

### 7.2 `SI00_inventory.md`

- SI08's inventory row rewritten to name the disposition and to add the wording
  ban to its "must not state" column.
- **New figure inventory**: all thirteen figures (four main, nine supporting)
  with state, evidence period, and the SI files each must agree with; a note that
  nine of them are POST-gated; a note that S9 is the only development-period
  figure and carries a mandatory scope band; and a note that figure
  cross-references are absent from the manuscript by design, per
  `FIGURE_REDRAW_SPEC.md` §7.
- Static source map: the `FIGURE_REDRAW_SPEC.md` row corrected from
  "Figure S1--S8" to "Figure S1--S9", and two rows added for the figure plan and
  the Stage-19 disposition.
- Required POST checks: item 6 corrected from `FigS1–FigS8` to `FigS1–FigS9`, and
  items 7 and 8 added for the SI08 disposition and the wording ban.

### 7.3 `si/README.md`

SI08's contents row rewritten to say "target-period projection, plus the
development-period non-reporting disposition"; the binder contract corrected from
`FigS1–FigS8` to `FigS1–FigS9`; and a prohibition added against writing "quantile
crossing".

### 7.4 SI file set

The SI00 inventory and the `si/README.md` contents table both list SI01–SI16.
The directory contains exactly `SI00_inventory.md` plus `SI01`–`SI16`. **No
mismatch.** The manuscript's Supporting Information paragraph names the same
sixteen in the same order.

---

## 8. Inconsistencies found, and how each was resolved

| # | Inconsistency | Resolution |
|---|---|---|
| 8.1 | `paper/FIGURE_REDRAW_SPEC.md` header said "four main-text figures and eight Supporting Information (SI) figures", while its own §1 state matrix and §5 specifications list S1–**S9** | Header corrected to nine, noting S9 is optional |
| 8.2 | Manuscript §8 said "Supporting figures S1–S8" | Rewritten to describe nine, grouped by evidence period |
| 8.3 | `SI00_inventory.md` and `si/README.md` referred to `FigS1–FigS8` in three places | All corrected to `FigS1–FigS9` |
| 8.4 | Manuscript §8 said primary observations "are redistributed from public providers", contradicting `FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3, where every relevant class defaults to `UNRESOLVED_EXCLUDE` | §8 rewritten; a retrieval-script-plus-manifest fallback is specified instead of asserted redistribution (§2.4) |
| 8.5 | The Key Points block violated AGU's 140-character limit at 201/157/201 characters | All three rewritten to 134/130/135 (§2.2) |
| 8.6 | SI08 did not distinguish the withheld development-period stage from the target-period trusted scorer | New SI08 §3 (§7.1) |
| 8.7 | `build_agu.py` would have raised on the rewritten Markdown (`REQUIRED_STATUS_TEXT`), and its hardcoded `KEYPOINTS` contradicted the Markdown | Generator fixed (§5.2) |
| 8.8 | `ThermoRoute_WRR.tex` is simultaneously **seal-valid** (hash matches the registry) and **content-stale** (superseded structure) | Cannot be fixed here; recorded in `agu_submission/README.md` and checklist items 7.3 and 8.4, which require regeneration *before* re-sealing |
| 8.9 | The checklist requires "figure/table references consistent with the 9-figure manifest" in the TeX, but `FIGURE_REDRAW_SPEC.md` §7 forbids inserting figure cross-references by hand and assigns them to the PRE/POST renderer | The in-repo spec was followed. No figure cross-reference was added anywhere. Recorded as checklist item 1.16 (**OPENING**) and stated in `agu_submission/README.md` so it is not mistaken for an oversight. |
| 8.10 | An early draft of the Wilks citation carried an unverified URL | Caught and replaced with an unlinked parenthetical before the file was finalised (§2.3) |

---

## 9. Placeholders left, and what must fill each

| Placeholder | File | Filled by |
|---|---|---|
| `[TO BE FILLED AFTER OPENING]` × 15 | manuscript §4.6 | The verified opening receipt via the deterministic renderer. Never by hand. |
| `[AUTHOR LIST TO BE COMPLETED]` | manuscript, generated TeX | Signed author intake |
| `[AFFILIATION 1 …]`, `[AFFILIATION 2 …]` | manuscript, generated TeX | Same; add or delete lines to match |
| `[CORRESPONDING AUTHOR TO BE COMPLETED]` | manuscript, cover letter, generated TeX | Same |
| `[INSTITUTIONAL E-MAIL TO BE COMPLETED]` | generated TeX | Same |
| `[FUNDING TO BE COMPLETED]` | manuscript Acknowledgments | Same |
| `[COMPUTATIONAL RESOURCES TO BE COMPLETED]` | manuscript Acknowledgments | Same |
| `[COMPETING INTERESTS TO BE COMPLETED]` | manuscript Acknowledgments | Same |
| `[CREDIT ROLES TO BE COMPLETED]` | manuscript Acknowledgments | Same, 14 CRediT terms |
| `[DATA DOI TO BE MINTED]` | manuscript §8 | Reviewed data deposit and minted DOI |
| `[DATA LICENCE TO BE ASSIGNED]` | manuscript §8 | Qualified byte-level rights decision |
| `[SOFTWARE DOI TO BE MINTED]` | manuscript §8 | Archived release deposit and minted DOI |
| `[RELEASE TAG TO BE ASSIGNED]` | manuscript §8 | Authors; `1.0.0` is not a deposited release |
| `[REPOSITORY URL TO BE CONFIRMED]` | manuscript §8 | Authors |
| `[DATE TO BE COMPLETED]` | cover letter | Authors, at submission |
| `[SUGGESTED REVIEWERS TO BE COMPLETED]` | cover letter | Authors |
| `[OPPOSED REVIEWERS TO BE COMPLETED]` | cover letter | Authors, or an explicit "none" |
| `[pending — 探索期数据，不可写入结论]` | SI06–SI16 | Their receipts, per the cell-level binder contract |

Every placeholder is in an unmistakable bracketed form. None resembles a real
identifier.

---

## 10. Deferred work

1. **Did not regenerate `ThermoRoute_WRR.tex`** — ~~`pandoc` absent and~~ the
   render guard refuses (§5.3). *Superseded 2026-08-05: pandoc was in fact
   available and the TeX has since been regenerated and compiled under
   `agujournal2025.cls`; see `docs/AGU2025_TEMPLATE_MIGRATION.md`. The render
   guard still refuses in-repo until §8 is re-sealed.*
2. **Did not re-seal `preopen_document_sha256`** — a `protocols/` change, outside
   the boundary. Four documents now need re-sealing together.
3. **Did not add figure cross-references** — forbidden by `FIGURE_REDRAW_SPEC.md`
   §7 (§8.9).
4. **Did not wire BibTeX into the generator** — a citation-convention decision
   for the authors (§5.4, checklist 6.4). *Partly superseded 2026-08-05: the
   generator now emits `\bibliography{../references}` and the reference list
   builds (41 entries, apacite). The in-text `\cite`/`\citeA` conversion is still
   an author decision.*
5. **Did not re-verify the bibliography against Crossref/DataCite** — no network,
   and `outputs/` is outside the write scope (§6.1).
6. **Did not materialise the Stage-19 measurement as a receipt-bound artifact** —
   it would require a committed script under `scripts/`. Still open as checklist
   item 8.5.
7. **Did not run pytest, anything under `scripts/`, or any git commit**, and did
   not write to either sibling worktree.

---

## 11. Note on concurrent activity

Concurrent working-tree changes were outside this revision and are not part of
the record above.
