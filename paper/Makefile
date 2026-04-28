# DermaJEPA paper build
#
#   make pdf       — compile main.tex to PDF (requires pdflatex + bibtex)
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
PYTHON   ?= uv run python
PYTHON_FIG ?= uv run --with matplotlib python

MAIN     := main
SECTIONS := $(wildcard sections/*.tex)
FIGURES  := figures/fig1-cross-backbone-gradient.pdf
TABLES   := $(wildcard tables/*.tex)

help:
	@echo "make pdf       — compile $(MAIN).tex to $(MAIN).pdf"
	@echo "make figures   — regenerate paper/figures/*.pdf"
	@echo "make verify    — verify numerical claims against local run mirror"
	@echo "make arxiv     — bundle arxiv.tar.gz"
	@echo "make clean     — remove LaTeX intermediates"

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) refs.bib $(FIGURES) $(TABLES)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX)   $(MAIN)
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex

figures: $(FIGURES)

figures/fig1-cross-backbone-gradient.pdf: figures/build_fig1.py figures/locked-numbers.json
	cd .. && $(PYTHON_FIG) paper/figures/build_fig1.py

verify:
	cd .. && $(PYTHON) paper/verify_claims.py

arxiv: pdf
	tar czf arxiv.tar.gz $(MAIN).tex $(SECTIONS) refs.bib $(FIGURES) $(TABLES)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf $(FIGURES) arxiv.tar.gz
