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

- fewer or more than eight structured `ROUTE_A_CLAIM` scope blocks;
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

The checked-in `ThermoRoute_WRR.tex` has been regenerated from the current
Markdown under `agujournal2025.cls` and compiles to a 44-page PDF with zero
errors and zero overfull boxes. It is *not* seal-valid: its bytes no longer match
`preopen_document_sha256`, which is the honest state — previously the file was
seal-valid and content-stale at the same time. It stays **not submittable** until
the author and opening items below are closed.

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
records `agujournal2019.cls` as `EXCLUDE_PUBLIC`: the evidence supports submission
use, not archive redistribution. The five files added for the 2025 migration
inherit exactly that status and have **no** provider-evidence row of their own
yet. Their SHA-256 values are listed in `docs/AGU2025_TEMPLATE_MIGRATION.md`.

They are also **not** in `scripts/verify_release.py`'s `ALLOWED_PAPER_MEMBERS`,
which is an exact allowlist for release archives. Any release archive built today
that includes them fails with "unregistered manuscript artifact"; any archive that
excludes them is not a compilable bundle. Resolving that is a `scripts/` +
rights-review change and is owner-owned.

`wiley-macros.tex` is shipped by AGU with the template but is **not** `\input` by
`agujournal2025.cls` in either branch. It is stored here for completeness only;
nothing in this build reads it.
