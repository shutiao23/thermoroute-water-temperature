# Build the AGU/WRR manuscript from hand-edited LaTeX.
#
# The .tex in this directory is the manuscript. It used to be generated from
# paper/ThermoRoute_paper.md by build_agu.py on every build, which meant any
# typesetting fix -- a float placement, a table that wanted \small, a manual
# line break in a heading -- was destroyed by the next conversion. Layout work
# is not expressible in Markdown, so the generator had to go once the paper
# reached the stage where layout mattered.
#
#   make            build the PDF
#   make watch      rebuild continuously while editing
#   make pages      print the page count (the submission cap is 23)
#   make clean      remove build products, keep the PDF
#   make distclean  remove the PDF too
#
# build_agu.py is retired but kept for provenance; see RETIRED_build_agu.md.
# It must not be run against ThermoRoute_WRR.tex again: it would overwrite
# hand-authored layout with a fresh conversion of the Markdown.

TEX     := ThermoRoute_WRR
PDF     := $(TEX).pdf

# AGU measures manuscript length in PUBLICATION UNITS, not pages:
#   PU = words/500 + figures + tables
# counted over the abstract, body, captions and appendices only -- the title,
# author block, plain-language summary, table interiors, Open Research,
# references and Supporting Information are all excluded.  The threshold is 25
# before excess-length fees.  `make units` is therefore the check that matters;
# the page count is reported for information because the submitted PDF is
# double-spaced (the class's `draft` option) and its page count says more about
# line spacing than about length.
PUCAP   := 25

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all watch pages units clean distclean check

all: $(PDF)

$(PDF): $(TEX).tex $(wildcard figures/*.pdf) $(wildcard *.cls) $(wildcard *.bib)
	$(LATEXMK) $(TEX).tex

watch:
	$(LATEXMK) -pvc $(TEX).tex

# Reported, not enforced: a double-spaced submission's page count is a fact
# about line spacing, not about whether the manuscript is too long.
pages: $(PDF)
	@n=$$(pdfinfo $(PDF) | awk '/^Pages:/ {print $$2}'); \
	 echo "$(PDF): $$n pages (double-spaced submission format)"

# This is the constraint AGU actually applies.
units:
	@python ../../scripts/final/count_publication_units.py --cap $(PUCAP)

check: units pages

clean:
	$(LATEXMK) -c $(TEX).tex
	rm -f $(TEX).bbl $(TEX).blg $(TEX).fls $(TEX).fdb_latexmk

distclean: clean
	rm -f $(PDF)
