# Pre-opening AGU / Water Resources Research typeset package

This directory uses the official `agujournal2019.cls` only as a typesetting
target. It is not yet a submission-ready package: empirical results are pending,
the author block is deliberately invalid, and repository/DOI/funding/license
metadata still require verified author input.

`../ThermoRoute_paper.md` is the only prose source. `build_agu.py` converts its
current abstract and numbered body, removes machine-only claim comments, supplies
status-safe AGU front matter, and refuses known withdrawn legacy claims. It does
not contain or infer numerical results.

## Rebuild and verify

```bash
python build_agu.py
python build_agu.py --check
pdflatex -interaction=nonstopmode -halt-on-error ThermoRoute_WRR.tex
pdflatex -interaction=nonstopmode -halt-on-error ThermoRoute_WRR.tex
```

The Markdown currently cites sources as linked author-year text, so the pre-opening
TeX does not depend on BibTeX. `references.bib` is retained as an archival candidate
and must be reconciled with in-text citations before submission.

## Required author actions before submission

1. Supply verified names, affiliations, ORCIDs, corresponding e-mail, funding,
   competing interests, repository URL, DOI, and redistribution/license language.
2. Complete the frozen computation and one-time evidence chain; regenerate the
   canonical Markdown only through the verified receipt and claim renderer.
3. Rebuild this TeX, run `--check`, compile it, and visually inspect every page.
4. Reconcile every in-text reference with the final bibliography and current AGU
   submission requirements.
