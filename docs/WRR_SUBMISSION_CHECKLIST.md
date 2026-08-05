# WRR submission checklist

| Field | Value |
|---|---|
| Target venue | Water Resources Research (AGU) |
| Manuscript | `paper/ThermoRoute_paper.md` (canonical prose source) |
| Branch | `feat/route-a-completion` |
| Date | 2026-08-05 |
| Overall state | **NOT SUBMITTABLE.** Every remaining item is listed below with the single external input that closes it. |

## How to read this file

Each item carries exactly one status:

| Status | Meaning |
|---|---|
| **DONE** | Complete in the repository as of this date. Nothing further is required. |
| **OPENING** | Blocked on the one-time 2021–2023 evaluation of manuscript §3.7. No human decision closes it; only the acquisition and the receipt-bound renderer do. |
| **AUTHORS** | Blocked on information only the authors can supply. Never invent it. |
| **DOI/RIGHTS** | Blocked on a deposit, a persistent identifier, or a qualified byte-level rights decision. |
| **PROTOCOL** | Blocked on an authorized change under `protocols/`, which is outside the manuscript-editing boundary. |
| **BUILD** | Blocked on a tool or environment that is absent here. |

An item is never marked DONE because it is "nearly" done.

---

## 1. Manuscript body

| # | Item | Status | What closes it |
|---|---|---|---|
| 1.1 | Title, restructured for WRR | DONE | — |
| 1.2 | Key Points: exactly three, each ≤ 140 characters, each a complete sentence | DONE | Mirrored byte-for-byte in `paper/highlights.md` and validated by `build_agu.py` |
| 1.3 | Abstract | DONE | — |
| 1.4 | Plain Language Summary | DONE | — |
| 1.5 | Keywords | DONE | — |
| 1.6 | §1–§3 Introduction, Data, Methods | DONE | — |
| 1.7 | §4.1–§4.5 development-period results, labelled exploratory | DONE | — |
| 1.8 | §4.6 evaluation-period result slots (15 `[TO BE FILLED AFTER OPENING]` markers across six tables) | **OPENING** | The verified opening receipt, rendered by `scripts/26_validate_claims.py --write-generated-results --require-complete`. Handwritten substitution is rejected by body hashes. |
| 1.9 | §5 Discussion, §6 Limitations, §7 Conclusions | DONE | — |
| 1.10 | Eight structured `ROUTE_A_CLAIM` scope blocks preserved byte-for-byte (5,982 bytes, SHA-256 `63698d1e…946cee`) | DONE | Verified 2026-08-05 |
| 1.11 | Both allowlisted legacy three-site sentences preserved verbatim, each appearing twice | DONE | Verified under whitespace normalisation |
| 1.12 | Registry `free_text_lints` (21 regexes) produce no unallowed hit | DONE | All three hits fall inside preserved claim blocks |
| 1.13 | B-02 forbidden-verb scan clean in manuscript, highlights, and cover letter | DONE | Every occurrence is prohibitive, inside a claim block, or a neutral non-claim use |
| 1.14 | Phrase "quantile crossing" absent everywhere | DONE | Enforced by `build_agu.py`; correct term is "zero-width (degenerate) nominal interval" |
| 1.15 | Manuscript status note replaced by the filled §4.6 | **OPENING** | Delete the "Manuscript status" block once §4.6 is filled; it describes a pre-opening state |
| 1.16 | Figure cross-references in the body | **OPENING** | `paper/FIGURE_REDRAW_SPEC.md` §7 fixes each first-citation position and forbids inserting them by hand. The PRE/POST renderer owns this. |
| 1.17 | Word count against the WRR limit | AUTHORS | Main body is ≈ 10,400 words (§1–§7). Confirm against the current WRR limit for the chosen article type, and trim §3 or §5 if required. |

## 2. Author and administrative metadata

| # | Item | Status | What closes it |
|---|---|---|---|
| 2.1 | Author list, order, and preferred citation names | **AUTHORS** | Signed intake, one row per author, per `docs/FAIR_SUBMISSION_READINESS_AND_TEMPLATES.md` §2 |
| 2.2 | Affiliations (the manuscript carries two placeholder affiliation lines; add or delete to match) | **AUTHORS** | Same intake |
| 2.3 | ORCID for each author | **AUTHORS** | Same intake. Journal-required ORCIDs must be present; any optional deposit ORCID must still be verified before use. |
| 2.4 | Corresponding author: verified name, ORCID, affiliation number, institutional e-mail | **AUTHORS** | Same intake |
| 2.5 | Funding statement with award numbers, or an explicit no-funding statement | **AUTHORS** | Same intake |
| 2.6 | Computational-resource acknowledgment | **AUTHORS** | Same intake |
| 2.7 | Competing-interests declaration from every author | **AUTHORS** | Same intake |
| 2.8 | CRediT contributor roles | **AUTHORS** | Same intake, 14 CRediT terms |
| 2.9 | Data-provider acknowledgments (USGS, ORNL DAAC, Northwest Knowledge Network) | DONE | Written, with an explicit statement that acknowledgment is not endorsement |
| 2.10 | Suggested and opposed reviewers | **AUTHORS** | `paper/cover_letter.md` |
| 2.11 | Cover letter date and signature | **AUTHORS** | `paper/cover_letter.md` |

## 3. Open Research / Data Availability Statement

AGU requires input data in a repository with a DOI, software archived with a DOI
and an explicit licence, and a clear statement for anything that cannot be
redistributed. Manuscript §8 now satisfies the *structure* of that policy; the
identifiers themselves are unminted.

| # | Item | Status | What closes it |
|---|---|---|---|
| 3.1 | Data-availability structure, primary sources with their own DOIs, and non-redistribution statements | DONE | Manuscript §8 |
| 3.2 | `[DATA DOI TO BE MINTED]` | **DOI/RIGHTS** | A deposit of the reviewed data members and a minted DOI |
| 3.3 | `[DATA LICENCE TO BE ASSIGNED]` | **DOI/RIGHTS** | Qualified byte-level rights decision. Every provider class currently defaults to `UNRESOLVED_EXCLUDE` per `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3. |
| 3.4 | `[SOFTWARE DOI TO BE MINTED]` | **DOI/RIGHTS** | Archived release deposit and minted DOI |
| 3.5 | `[RELEASE TAG TO BE ASSIGNED]` | **AUTHORS** | Package version `1.0.0` is not a deposited release |
| 3.6 | `[REPOSITORY URL TO BE CONFIRMED]` | **AUTHORS** | Repository visibility is a fact, not a redistribution authorization |
| 3.7 | Software licence identifier (MIT, SPDX `MIT`) | DONE | Stated in §8 from the repository `LICENSE` and `pyproject.toml`, with the explicit scope limit that MIT covers source code only |
| 3.8 | `THIRD_PARTY_NOTICES` register | **DOI/RIGHTS** | Absent. Required before any archive member list is final. |
| 3.9 | AGU class file disposition | DONE (as a statement) | §8 states it is supplied for submission and not redistributed; the matrix records it as `EXCLUDE_PUBLIC` |
| 3.10 | Legacy three-station CSV disposition | DONE | §2.5 and §8 state they are not redistributed in any form |
| 3.11 | Retrieval-script-plus-manifest fallback for non-redistributable provider bytes | DONE (as a specification) | §8 specifies it; the archive that implements it does not yet exist |
| 3.12 | `CITATION.cff`, `codemeta.json`, deposit metadata | **DOI/RIGHTS** | Deliberately absent. Do not create with placeholder creators or a draft DOI. |
| 3.13 | Public archive build | **DOI/RIGHTS** | Currently fail-closed by design; `LOCAL_EVIDENCE_ONLY` is the only enabled distribution state |
| 3.14 | Independent clean-host reproduction | **DOI/RIGHTS** | §8 states plainly that it has not been carried out |

## 4. Supporting Information

| # | Item | Status | What closes it |
|---|---|---|---|
| 4.1 | SI00 inventory matches the actual file set (SI00 plus SI01–SI16) | DONE | — |
| 4.2 | SI00 carries a figure inventory consistent with the 9-figure POST manifest | DONE | Added 2026-08-05 |
| 4.3 | SI08 corrected for the Stage-19 disposition, carrying the measured table verbatim | DONE | — |
| 4.4 | SI01–SI05 static design content | DONE | — |
| 4.5 | SI06–SI16 receipt projections | **OPENING** | Each is an empty projection until its receipt exists |
| 4.6 | SI15 reproduction tolerances, commands, and expected digests | **OPENING** | Manuscript §8 promises these; they must exist before submission |
| 4.7 | SI filenames and order match the manuscript's Supporting Information paragraph | DONE | — |

## 5. Figures

| # | Item | Status | What closes it |
|---|---|---|---|
| 5.1 | Figures 1, S1, S2, S3 materialized as PRE artifacts | DONE | PDF, SVG, PNG present under `paper/*/figures/` |
| 5.2 | Figures 2, 3, 4, S4–S8 rendered | **OPENING** | All blocked on the opening receipt and the nine `trusted/` artifacts |
| 5.3 | Figure S9 (optional development-period sensitivity) | **OPENING** | Its substantive evidence already resolves; blocked only on the spec state and the opening receipt. Droppable without weakening a registered claim. |
| 5.4 | Pre-opening degeneracy check on target-period predictions before the opening runs | **OPENING** | A read-only check that guards a one-shot operation against an abort. Costs minutes; run it first. |
| 5.5 | Figure captions for Figures 2–4 and S4–S9 | **OPENING** | Captions bind value IDs that do not yet exist |
| 5.6 | Render profile compliance (vector, embedded fonts, AGU widths 85 mm / 140 mm, colour-blind-safe, redundant non-colour encoding) | DONE (as a control) | Enforced in code by `RenderProfile.validate()` |
| 5.7 | Figure spec header corrected from "eight SI figures" to nine | DONE | 2026-08-05 |

## 6. Bibliography

| # | Item | Status | What closes it |
|---|---|---|---|
| 6.1 | Every in-text citation resolves to an entry in `paper/references.bib` | DONE | Verified 2026-08-05; 41 entries |
| 6.2 | Every entry in `paper/references.bib` is cited in the manuscript | DONE | One uncited entry removed (the R `dataRetrieval` package; the acquisition uses the Python client) |
| 6.3 | No invented DOI, page range, or identifier | DONE | `url` fields were added only where the manuscript already carried that exact link |
| 6.4 | Convert in-text linked author-year prose to `\cite`/`\citeA` keys | **AUTHORS** | Partly closed 2026-08-05: `build_agu.py` now emits `\bibliography{../references}` and the reference list builds (41 entries, apacite, bibtex 0 warnings, 0 undefined citations). No `.bst` needs vendoring — `agujournal2025.cls` sets `\bibliographystyle{apacite}` itself. Still open: the prose carries no `\cite`/`\citeA` key, so the generator emits `\nocite{*}` as an explicit bridge, sound only because 6.1/6.2 reconciled the `.bib` 1:1 with the in-text list. **Delete `\nocite{*}` in the same change that adds real keys.** |

## 7. AGU TeX package

| # | Item | Status | What closes it |
|---|---|---|---|
| 7.1 | `build_agu.py` reads the rewritten Markdown, extracts Key Points, Plain Language Summary, and keywords, and routes AGU back matter | DONE | Extraction and validation verified against the current Markdown |
| 7.2 | Generator refuses on lost result-slot markers, wrong Key Point count or length, banned phrases, withdrawn claims | DONE | — |
| 7.3 | `paper/agu_submission/ThermoRoute_WRR.tex` regenerated | DONE (bytes) / **PROTOCOL** (seal) | Regenerated 2026-08-05 from the current Markdown under `agujournal2025.cls`. The `pandoc`-absent half of the old blocker was **wrong**: pypandoc vendors the binary off `PATH` and `_pandoc_path()` already finds it — use `/home/lzq/anaconda3/envs/route-a/bin/python`. The render-guard half stands: the file is now content-fresh but **not seal-valid**, which is the honest state (it was previously seal-valid and content-stale). Closes on 8.1–8.4. |
| 7.4 | `build_agu.py --check` passes | **PROTOCOL** | Verified passing against a shadow tree carrying the to-be-sealed hashes (`AGU TeX is current and contains no withdrawn claim`), byte-identical to the checked-in file. Fails in-repo until 8.1–8.3 are re-sealed. |
| 7.5 | PDF compiles cleanly | DONE | `latexmk -pdf` converges in 3 passes + bibtex: **44 pages**, US Letter, **0 errors, 0 overfull boxes, 0 undefined references**. Remaining warnings are class-inherited and enumerated in `docs/AGU2025_TEMPLATE_MIGRATION.md` §6. |
| 7.8 | Migrated from the superseded `agujournal2019.cls` to `agujournal2025.cls` | DONE | 2026-08-05. Front-matter order, `\keypoints{}{}{}`, `\authoraddr` + `\correspondingauthor`, nested `plainlanguagesummary`, `\maketitle`, running heads, `\journalname` after `\begin{document}`, Open Research **Statement**. Full record: `docs/AGU2025_TEMPLATE_MIGRATION.md`. |
| 7.9 | Open Research Statement positioned immediately before the references | **AUTHORS** | The template requires it there; the manuscript has Acknowledgments and Supporting Information in between. The generator does not reorder narrative sections. |
| 7.10 | Vendored 2025 class assets registered for release and rights | **DOI/RIGHTS** + `scripts/` | `agujournal2025.cls`, `tweaklist-git-moderncv-fixed.sty`, `wiley-macros.tex`, and both logo PDFs are third-party bytes with no provider-evidence row and are absent from `verify_release.py`'s exact `ALLOWED_PAPER_MEMBERS`. An archive including them fails; one excluding them will not compile. SHA-256s in `docs/AGU2025_TEMPLATE_MIGRATION.md` §4. |
| 7.6 | Author block, affiliations, corresponding author in the TeX | **AUTHORS** | Generator emits explicit placeholders and does not invent them |
| 7.7 | POST submission renderer (result-bearing TeX/PDF) | **OPENING** | Does not exist. PRE-only `--check` is not a POST publication check. |

## 8. Protocol seals

| # | Item | Status | What closes it |
|---|---|---|---|
| 8.1 | `preopen_document_sha256` for `paper/ThermoRoute_paper.md` | **PROTOCOL** | Rewritten 2026-08-05; hash no longer matches |
| 8.2 | `preopen_document_sha256` for `paper/highlights.md` | **PROTOCOL** | Rewritten 2026-08-05; hash no longer matches |
| 8.3 | `preopen_document_sha256` for `paper/cover_letter.md` | **PROTOCOL** | Rewritten 2026-08-05; hash no longer matches |
| 8.4 | `preopen_document_sha256` for `paper/agu_submission/ThermoRoute_WRR.tex` | **PROTOCOL** | 7.3 has now regenerated it, so the hash no longer matches (was `34a34af4…`, now `de84e612…`). Re-seal with 8.1–8.3. Note the registry is itself bound through `MODEL_MATRIX_CLAIM_REGISTRY_SHA256`, so the re-seal is one governed batch — see `docs/M01_M14_M15_MANUSCRIPT_BATCH_DEFERRED.md`. |
| 8.5 | Stage-19 measurement materialized as a receipt-bound artifact | **PROTOCOL** | Currently derived by a read-only ad hoc measurement; re-derive from a committed script and record path and digest |
| 8.6 | Per-station LightGBM development score | **OPENING** | No scored summary exists for that variant; either score it or state its absence in SI07 |

Re-sealing is a `protocols/` change and creates a new document lineage. Items
8.1–8.4 must be re-sealed together, after 7.3, so that all four documents enter
the same lineage. The superseded bytes of 8.1–8.3 remain recoverable at commit
`b0699a8`.

## 9. Journal submission mechanics

| # | Item | Status | What closes it |
|---|---|---|---|
| 9.1 | Article type and section selection in GEMS | **AUTHORS** | — |
| 9.2 | Manuscript PDF and source files | **BUILD** | Requires 7.3–7.5 |
| 9.3 | Supporting Information as a single PDF or per AGU's current SI format | **OPENING** | Requires §4 |
| 9.4 | Figures as separate files at AGU's required resolution and format | **OPENING** | Requires §5 |
| 9.5 | Cover letter | **AUTHORS** | Written; requires 2.10, 2.11, and the send condition at its head |
| 9.6 | Availability statement pasted into the submission form | **DOI/RIGHTS** | Must match manuscript §8 exactly |
| 9.7 | Preprint policy and prior-submission declaration | **AUTHORS** | — |

---

## Critical path

1. Run the read-only pre-opening degeneracy check on the target-period
   predictions (5.4). It is cheap and guards a one-shot, multi-day-to-recover
   operation.
2. Execute the one-time acquisition and scoring (§3.7), producing the opening
   receipt and the nine `trusted/` artifacts.
3. Render §4.6 and SI06–SI16 from the receipt; render Figures 2–4 and S4–S9.
4. Collect signed author administration (§2) in parallel with steps 1–3; it does
   not depend on any of them.
5. Close the rights review and mint both DOIs (§3). This is the longest external
   dependency and should start first.
6. ~~Install `pandoc`,~~ re-seal the four frozen documents (§8), ~~regenerate the
   TeX, and compile~~. *2026-08-05: pandoc was already available; the TeX is
   regenerated under `agujournal2025.cls` and compiles to a 44-page PDF. Only the
   re-seal remains, and it is now a single governed batch (8.1–8.4).*
7. Convert the citation convention to `\cite`/`\citeA` keys and drop the
   `\nocite{*}` bridge (6.4). The reference list itself already builds.

Items 4 and 5 have no technical dependency on the opening and are the two things
that can be started today.
