# ThermoRoute — AGU / Water Resources Research submission

Official AGU LaTeX template (`agujournal2019.cls`, downloaded from the AGU
`myst-templates/agu2019` GitHub `original/` folder) with the ThermoRoute
manuscript filled in.

## Files
- `ThermoRoute_WRR.tex` — the manuscript in AGU macros (`\journalname`, `\authors`,
  `\keypoints`, `\begin{abstract}`, `\cite`/`\citeA`, `\bibliography`).
- `references.bib` — 41 web-verified references (apacite/BibTeX).
- `agujournal2019.cls`, `trackchanges.sty` — official AGU class + style.
- `agujournaltemplate.tex` — the pristine AGU template (reference only).
- `figures/` — the four manuscript figures.
- `build_agu.py` — regenerates `ThermoRoute_WRR.tex` from
  `../ThermoRoute_paper.md` (single source of truth for content).

## Compile
```
pdflatex ThermoRoute_WRR
bibtex   ThermoRoute_WRR
pdflatex ThermoRoute_WRR
pdflatex ThermoRoute_WRR
```
Produces `ThermoRoute_WRR.pdf` (19 pp). Also compiles unchanged on Overleaf:
upload this folder and set `ThermoRoute_WRR.tex` as the main document.

## Before submission (author actions)
1. Replace `[Author One/Two/Three]`, affiliations, and the corresponding e-mail
   in `ThermoRoute_WRR.tex` (front matter).
2. Optional: increase inline citation density — the body currently cites a few
   works with `\citeA`/`\cite` and pulls the full list via `\nocite{*}`; WRR
   reviewers prefer each reference cited at its point of use.
3. The single model-prior equation is set as a real `align` block; the remaining
   pseudo-equations are inline `\texttt{}` — promote to display math if desired.

Content stays in `../ThermoRoute_paper.md`; re-run `build_agu.py` after edits.
