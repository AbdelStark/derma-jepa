# DermaJEPA paper build
#
#   make pdf       — verify claims and compile main.tex to PDF
#   make figures   — regenerate paper/figures/*.pdf from build_fig*.py
#   make tables    — verify paper/tables/*.tex compile standalone (no-op if
#                    they include only LaTeX, no Python builders for now)
#   make verify    — run paper/verify_claims.py against the local run mirror
#   make arxiv     — produce arxiv.tar.gz with sources + figures
#   make clean     — remove LaTeX intermediates
#   make distclean — also remove generated figures and the arxiv tarball

.PHONY: pdf figures tables verify arxiv clean distclean help

PDFLATEX ?= pdflatex
BIBTEX   ?= bibtex
TECTONIC ?= tectonic
PYTHON   ?= uv run python
PYTHON_FIG ?= uv run --with matplotlib python

MAIN     := main
SECTIONS := $(wildcard sections/*.tex)
FIGURES  := figures/fig1-cross-backbone-gradient.pdf \
            figures/fig2-train-vs-test-scatter.pdf \
            figures/fig3-pair-score-histograms.pdf
TABLES   := $(wildcard tables/*.tex)

help:
	@echo "make pdf       — compile $(MAIN).tex to $(MAIN).pdf"
	@echo "make figures   — regenerate paper/figures/*.pdf"
	@echo "make verify    — verify numerical claims against local run mirror"
	@echo "make arxiv     — bundle arxiv.tar.gz"
	@echo "make clean     — remove LaTeX intermediates"

pdf: verify $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) refs.bib $(FIGURES) $(TABLES)
	@if command -v $(TECTONIC) >/dev/null 2>&1; then \
		$(TECTONIC) --keep-intermediates --keep-logs $(MAIN).tex; \
	elif command -v $(PDFLATEX) >/dev/null 2>&1; then \
		$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex; \
		$(BIBTEX)   $(MAIN); \
		$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex; \
		$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex; \
	else \
		echo "No TeX engine found: install tectonic or pdflatex."; \
		exit 127; \
	fi

figures: $(FIGURES)

figures/fig1-cross-backbone-gradient.pdf: figures/build_fig1.py figures/locked-numbers.json
	cd .. && $(PYTHON_FIG) paper/figures/build_fig1.py

figures/fig2-train-vs-test-scatter.pdf: figures/build_fig2.py
	cd .. && $(PYTHON_FIG) paper/figures/build_fig2.py

figures/fig3-pair-score-histograms.pdf: figures/build_fig3.py
	cd .. && $(PYTHON_FIG) paper/figures/build_fig3.py

verify:
	cd .. && $(PYTHON) paper/verify_claims.py

arxiv: verify pdf
	# arXiv compiles without bibtex; include main.bbl so cross-refs resolve.
	COPYFILE_DISABLE=1 tar czf arxiv.tar.gz $(MAIN).tex $(MAIN).bbl $(SECTIONS) refs.bib \
	                     $(FIGURES) $(TABLES)
	@echo "arxiv.tar.gz contents:"
	@tar tzf arxiv.tar.gz | sort
	@tar tzf arxiv.tar.gz | awk -F/ '$$NF ~ /^\\._/ { print "AppleDouble file in archive: " $$0; bad=1 } END { exit bad }'

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf $(FIGURES) arxiv.tar.gz
