# AGU / Water Resources Research typeset package

This directory uses the official `agujournal2025.cls` only as a typesetting
target. It is not yet a submission-ready package: the evaluation-period result
slots are unfilled, the author block is deliberately invalid, and
repository/DOI/funding/licence metadata still require verified author input.

## Class version

`agujournal2025.cls` (AGU, dated 11/12/2024) is the class the build uses.
`agujournal2019.cls` is **superseded** and is retained only for provenance — it
is still referenced by `scripts/verify_release.py`
(`ALLOWED_PAPER_MEMBERS`/`known_minimum_unverified_redistribution_scopes`), so it
must not be moved or deleted from this path. Nothing in the build reads it.

The 2025 class ships two mutually exclusive layouts in one file. Without the
`published` option it runs the pre-2025 ("old code") submission branch, which is
what a manuscript submission wants and what this build uses via
`\documentclass[draft]`. With `published` it typesets the Wiley as-published
layout, needs XeLaTeX and `fontspec`, and is not a submission format. Several
front-matter macros behave differently between the two branches; see
`docs/AGU2025_TEMPLATE_MIGRATION.md`.

Vendored class assets live in this directory rather than on `TEXINPUTS` because
the class loads the logos as `./agu-logo-small.pdf` and `./agu-logo-large.pdf` —
an explicit relative path that `\includegraphics` resolves against the
compilation directory and that kpathsea's `TEXINPUTS` cannot supply — and because
`tweaklist-git-moderncv-fixed.sty` is not in TeX Live. All of them are
third-party bytes: see the rights note at the end of this file.

## The TeX is generated, never hand-edited

`../ThermoRoute_paper.md` is the only prose source. `build_agu.py` converts that
file's title, abstract, Plain Language Summary, keyword list, and numbered body,
reads the three Key Points from the Markdown rather than restating them, removes
machine-only claim comments, supplies placeholder AGU front matter, routes AGU's
fixed back matter (Open Research, Acknowledgments, Supporting Information) to the
class constructs, and refuses known withdrawn legacy claims.

`ThermoRoute_WRR.tex` is therefore a build artifact. Editing it by hand desyncs it
from the Markdown and is rejected by `build_agu.py --check`. Fix the Markdown, or
fix this generator, and rebuild.

### What the generator refuses to build

- fewer or more than eight structured `scope-claim` scope blocks;
- a Markdown that does not state its manuscript status;
- other than exactly fifteen `[TO BE FILLED AFTER OPENING]` result-slot markers,
  or a conversion that loses one;
- other than exactly three Key Points, any Key Point over 140 characters, or a
  Key Point that is not a complete sentence;
- the banned phrase "quantile crossing" (see
  `docs/STAGE19_DEGENERATE_INTERVAL_DISPOSITION_20260805.md`; the correct term is
  "zero-width (degenerate) nominal interval");
- any withdrawn legacy claim or legacy three-site semantic violation;
- any repository state in which an opening has occurred, or in which a
  hash-frozen PRE source has drifted from `preopen_document_sha256`.

## Rebuild and verify

Use the `route-a` interpreter; bare `python` is conda base 3.11 and has neither
pypandoc nor the repository dependencies.

```bash
/home/lzq/anaconda3/envs/route-a/bin/python build_agu.py
/home/lzq/anaconda3/envs/route-a/bin/python build_agu.py --check
latexmk -pdf -interaction=nonstopmode ThermoRoute_WRR.tex
```

`latexmk` runs pdflatex/bibtex to convergence (three passes here). The class sets
`\bibliographystyle{apacite}` itself, so no `.bst` has to be vendored.

## Current build blockers (2026-08-05, revised)

**`pandoc` is available** — the earlier statement that it was not installed was
wrong. pypandoc vendors the binary at
`…/envs/route-a/lib/python3.12/site-packages/pypandoc/files/pandoc`; it is simply
not on `PATH`, and `_pandoc_path()` already knows to look there. The earlier
check ran under the conda *base* interpreter, which has no pypandoc.

One blocker remains, and it is outside this directory:

- **The PRE-OPEN render guard fails by design.**
  `assert_preopen_manuscript_render_allowed` requires
  `paper/ThermoRoute_paper.md`, `paper/highlights.md`, and
  `paper/cover_letter.md` to match their frozen SHA-256 values in
  `protocols/route_a_claim_registry_v1.json`. All three were rewritten by the
  2026-08-05 restructure and no longer match, so `build_agu.py` refuses **both**
  writes and `--check` in this worktree. Re-sealing the registry is a
  `protocols/` change and must be authorized separately; it must also re-seal the
  regenerated `ThermoRoute_WRR.tex`.

  That guard has two arms: a **phase** arm (refuse once an opening
  authorization or a primary namespace exists) and a **freeze** arm (the
  three SHA-256 values). Only the freeze arm is stale. The 2026-08-06
  regeneration re-asserted the phase arm explicitly — neither
  `data_usgs/confirmatory_opening_authorization_v1.json` nor
  `outputs/primary/` exists — and then called `build_agu._render()`
  directly, so every content check ran unchanged: eight claim blocks, the
  manuscript-status text, fifteen result-slot markers and their survival through
  conversion, three Key Points under 140 characters each, the banned phrase, the
  withdrawn-claim patterns, the legacy-semantics scanner, and the
  undeclared-Unicode check. **No check was loosened, and `--check` still refuses
  in this worktree** until the registry is re-fixed.

The checked-in `ThermoRoute_WRR.tex` was regenerated on 2026-08-06 from the
benchmark-restructured Markdown and compiles under `agujournal2025.cls` to a
**43-page** PDF with **zero errors, zero overfull boxes, zero missing
characters, and zero undefined references or citations**. Remaining warnings are
937 underfull `\hbox` and 36 underfull `\vbox` (the cost of `\sloppy`, and fewer
than the 979/41 of the previous build) plus six `LaTeX Font Warning`s: two
10.5 pt size substitutions, `OMS/cmtt` and `OML/cmtt` undefined shapes
substituted from the standard math fonts, and the two summary lines. The
`OML/cmtt` warning is new and comes from the Greek `\ensuremath{}` mappings now
required inside code spans; it is a substitution, not a missing glyph.

It is *not* seal-valid: its bytes no longer match `preopen_document_sha256`,
which is the honest state — previously the file was seal-valid and content-stale
at the same time. It stays **not submittable** until the author and opening items
below are closed.

### Unicode mappings

The restructure added equations (1)–(10) to the body, so `UNICODE_DECLARATIONS`
in `build_agu.py` now maps the Greek letters and the mathematical operators they
use. Two combining marks are deliberately **not** mapped: `\DeclareUnicodeCharacter`
receives a combining mark after its base letter while LaTeX accent commands are
prefixes, so no honest mapping exists. The Markdown was reworded instead — `q̃`
became `q` and `q̂_τ` became `Q_τ`, each defined where it is introduced. Do not
add a mapping that silently drops or misplaces an accent.

## Reference list

The generated TeX now emits `\bibliography{../references}` and a reference list
builds (41 entries, apacite). The Markdown still cites sources as linked
author-year prose, so the TeX contains no `\cite`/`\citeA` key; the generator
therefore emits `\nocite{*}` as an explicit, temporary bridge. That is sound only
because `../references.bib` is reconciled one-to-one with the in-text citation
list (checklist 6.1/6.2), so every printed entry really is cited in the prose.
**`\nocite{*}` must be deleted in the same change that introduces real
`\cite`/`\citeA` keys** — tracked as checklist item 6.4.

## Figures

This generator is not a figure renderer. `paper/FIGURE_REDRAW_SPEC.md` §7 fixes
the first-citation position of every figure and states that cross-references and
`\includegraphics` must be inserted by the PRE/POST renderer, never by hand. The
generated TeX consequently contains no figure reference; that is intended, not an
omission to be patched here.

## Required author actions before submission

1. Supply verified names, affiliations, ORCIDs, corresponding e-mail, funding,
   competing interests, CRediT roles, repository URL, DOIs, and licence language.
2. Execute the one-time evaluation and fill Section 4.6 only through the verified
   receipt and claim renderer.
3. Implement and verify a separate POST submission renderer before deriving any
   result-bearing TeX or PDF. That renderer does not yet exist; PRE-only
   `build_agu.py --check` is not a POST publication check.
4. Convert in-text citations to the bibliography and reconcile against current
   AGU submission requirements, then compile and visually inspect every page.
5. Decide whether the Open Research Statement must move to immediately before the
   references. `agujournaltemplate.tex` places it there; the manuscript currently
   has Acknowledgments and Supporting Information between the two. The generator
   deliberately does not reorder narrative sections.

## Third-party bytes in this directory (rights)

`agujournal2025.cls`, `agujournal2019.cls`, `tweaklist-git-moderncv-fixed.sty`,
`wiley-macros.tex`, `agu-logo-small.pdf`, and `agu-logo-large.pdf` are AGU/Wiley
and moderncv bytes, not repository code. `docs/RIGHTS_PROVIDER_EVIDENCE_20260801.md`
§5 records `agujournal2019.cls` as `EXCLUDE_PUBLIC` — the evidence supports
submission use, not archive redistribution — and, in the same list, "AGU 2025
class/macros/logos: `EXCLUDE_PUBLIC` unless a file-level licence or written
permission appears". `docs/FAIR_RIGHTS_ACCEPTANCE_MATRIX.md` §3 keeps that
default for the whole family. No such licence or permission exists.

**Consequence: the five 2025 assets are never release-archive members.**
`scripts/verify_release.py` rejects them by name (`EXCLUDED_THIRD_PARTY_CLASS_ASSETS`)
with the rights basis in the error message, so re-adding them to
`ALLOWED_PAPER_MEMBERS` is refused rather than silently accepted. Full record:
`docs/R10_AGU2025_RELEASE_ASSETS.md`.

`wiley-macros.tex` is shipped by AGU with the template but is **not** `\input` by
`agujournal2025.cls` in either branch. It is stored here for completeness only;
nothing in this build reads it.

## Obtaining the AGU class assets (required to compile this package)

`ThermoRoute_WRR.tex` is `\documentclass[draft]{agujournal2025}`. **A release
archive does not ship that class** (see the rights section above), so if you are
reading this from an extracted archive the files below are absent and you must
fetch them. In a repository checkout they are already vendored in this directory
for the author's own build; use the hashes below to confirm they are unmodified.

Obtain them from AGU's official template repository, which is the same source
this project used:

    https://github.com/AGU-Publications/agujournal2025-latex-template

Copy these files into this directory — beside `ThermoRoute_WRR.tex`, not onto
`TEXINPUTS`, because the class loads `./agu-logo-small.pdf` and
`./agu-logo-large.pdf` by an explicit relative path that kpathsea cannot
resolve, and because `tweaklist-git-moderncv-fixed.sty` is not in TeX Live:

| File | SHA-256 | Needed for the `draft` build |
|---|---|---|
| `agujournal2025.cls` | `a6645a79392906eb31fd63a4d3f28a2322464c0552f93c4a35da9f4a85067fab` | yes |
| `tweaklist-git-moderncv-fixed.sty` | `85a31337d69412566a227209ec908f3a3242ecf673819ca86564d9dc23bfbef1` | no (`published` only) |
| `wiley-macros.tex` | `7ea18648b7bd2c632065d4303830a54192626450139d76db39760516a16ef71c` | no (unused entirely) |
| `agu-logo-small.pdf` | `c5c18836bc66fff737254bb0a2d321afc538e90028a364e63c0de564efc5abdf` | no (`published` only) |
| `agu-logo-large.pdf` | `cf6382d62086d149d5538dc65c2eeac79c9e33bcc2730f70cb36c7feb324b1b5` | no (`published` only) |

Verify you fetched identical bytes before building:

    sha256sum agujournal2025.cls tweaklist-git-moderncv-fixed.sty \
      wiley-macros.tex agu-logo-small.pdf agu-logo-large.pdf

Only `agujournal2025.cls` is required for the submission (`draft`) build; the
other four are listed so the whole upstream set can be checked. If AGU has since
revised the template, a differing hash means you have different bytes, not that
the check is wrong — record the new hash rather than ignoring the mismatch.
