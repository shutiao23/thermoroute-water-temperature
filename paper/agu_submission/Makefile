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
PAGECAP := 23

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all watch pages clean distclean check

all: $(PDF)

$(PDF): $(TEX).tex $(wildcard figures/*.pdf) $(wildcard *.cls) $(wildcard *.bib)
	$(LATEXMK) $(TEX).tex

watch:
	$(LATEXMK) -pvc $(TEX).tex

# The page count is a submission constraint, so it is a build target rather
# than something to remember to check.
pages: $(PDF)
	@n=$$(pdfinfo $(PDF) | awk '/^Pages:/ {print $$2}'); \
	 echo "$(PDF): $$n pages (cap $(PAGECAP))"; \
	 if [ "$$n" -gt "$(PAGECAP)" ]; then \
	   echo "OVER by $$(( $$n - $(PAGECAP) )) pages"; exit 1; \
	 fi

check: pages

clean:
	$(LATEXMK) -c $(TEX).tex
	rm -f $(TEX).bbl $(TEX).blg $(TEX).fls $(TEX).fdb_latexmk

distclean: clean
	rm -f $(PDF)
