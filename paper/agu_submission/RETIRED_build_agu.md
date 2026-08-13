# `build_agu.py` is retired — the LaTeX is now the manuscript

**Do not run `build_agu.py` against `ThermoRoute_WRR.tex`.** It regenerates the
file from `paper/ThermoRoute_paper.md` and will silently destroy any typesetting
work. Build with `make` in this directory.

## Why it existed

The generator enforced something real. Markdown was the single prose source, the
converter contained no empirical sentence of its own, a list of withdrawn claims
was rejected before any TeX was written, and `--check` failed CI when the
checked-in `.tex` was not byte-identical to what the source would produce. That
made "the PDF says what the manuscript says" a machine-checked property rather
than a habit, and during the period when numbers were being withdrawn and
restored it caught drift that a human diff would not have.

## Why it had to go

A generated file cannot be typeset. Everything a paper needs at submission —
float placement, a table set `\small` so it fits a column, a manual break in a
heading, a figure moved across a page boundary, a `\vspace` that closes a
one-line overflow — lives in the LaTeX and is expressible in no Markdown. Under
the generator every such fix survived until the next conversion and then
vanished, with no error, because regenerating was the normal operation. Cutting
this manuscript from 58 pages to 23 is almost entirely layout work of that kind,
so the generator stopped being a safety property and became the main obstacle to
the work.

## What replaces the guarantee

The generator's real contribution was not the conversion; it was three checks
that happened to be bolted to it. Those move rather than disappear:

- **Withdrawn numbers and prohibited phrases.** `check_manuscript_consistency.py`
  already scans the `.tex` alongside the Markdown (`documents["tex"]`), so the
  quarantine gate covers the manuscript in its authored form.
- **Claims match their artifacts.** The claim ledger resolves every headline
  number against a scored artifact under `outputs/final/`, independently of how
  the document was produced.
- **Post-outcome promotion.** `POST_OUTCOME_PROMOTED` requires the Abstract and
  Key Points to disclose the status of the promoted information results; the
  build fails if the disclosure is deleted.

What is genuinely lost is byte-exact Markdown/TeX correspondence. That is the
price of being able to typeset, and it is paid deliberately: from now on the
`.tex` is authoritative and `ThermoRoute_paper.md` is a historical draft, no
longer the source of record. It is kept because the decision log and the SI
cross-reference it, not because it drives anything.

## Final conversion

The `.tex` was regenerated from the Markdown one last time immediately before
retirement, so the handover point is a clean conversion rather than a partially
hand-edited file. Everything after that commit is authored LaTeX.

## If you need the old behaviour

`build_agu.py` still runs and still works against a *different* output path.
Point it somewhere other than `ThermoRoute_WRR.tex` — the CI step that ran
`--check` has been removed, so nothing enforces the correspondence any more and
a regeneration would be a silent overwrite rather than a checked one.
