# AGU / Water Resources Research typeset package

This directory uses the official `agujournal2019.cls` only as a typesetting
target. It is not yet a submission-ready package: the evaluation-period result
slots are unfilled, the author block is deliberately invalid, and
repository/DOI/funding/licence metadata still require verified author input.

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

```bash
python build_agu.py
python build_agu.py --check
pdflatex -interaction=nonstopmode -halt-on-error ThermoRoute_WRR.tex
pdflatex -interaction=nonstopmode -halt-on-error ThermoRoute_WRR.tex
```

## Current build blockers (2026-08-05)

The checked-in `ThermoRoute_WRR.tex` still carries the **superseded** manuscript
structure (old title, old abstract, sections "Problem and scope" through
"Conclusion"). It cannot be regenerated in the present state for two independent
reasons, both outside this directory:

1. **`pandoc` is not installed** in the working environment, and neither is
   `pypandoc`. `build_agu.py` requires one of them.
2. **The PRE-OPEN render guard fails by design.** `assert_preopen_manuscript_render_allowed`
   requires `paper/ThermoRoute_paper.md`, `paper/highlights.md`, and
   `paper/cover_letter.md` to match their frozen SHA-256 values in
   `protocols/route_a_claim_registry_v1.json`. All three were rewritten and no
   longer match. Re-sealing the registry is a `protocols/` change and must be
   authorized separately; it will also need to re-seal the regenerated
   `ThermoRoute_WRR.tex`, whose own frozen hash still matches its stale bytes.

Until both are resolved, treat `ThermoRoute_WRR.tex` as **stale and not
submittable**. Its bytes are retained only so the eventual regeneration has a
diff baseline.

## Reference list

The Markdown cites sources as linked author-year text, so the generated TeX
contains no `\cite` and no `\bibliography`, and therefore renders **no reference
list**. `../references.bib` is reconciled one-to-one with the in-text citation
list, but converting the citation convention to `\cite`/`\citeA` keys against it
(class default bibliography style: `apacite`) is a required pre-submission step
and is tracked in `docs/WRR_SUBMISSION_CHECKLIST.md`.

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
