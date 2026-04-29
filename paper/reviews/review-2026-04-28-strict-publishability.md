# Strict Academic Publishability Review

Paper: `paper/main.tex`

Repository state reviewed: `4292d44` on `main`, then updated locally in this
pass.

Date: 2026-04-28

## Verdict

Ready for a preliminary arXiv-style technical report or feedback-seeking
workshop submission.

Still not ready for a strong full peer-reviewed ML / medical-imaging venue
claim without EXP-009 or an equivalent partition experiment. The manuscript now
states the central caveat correctly, but the DermLIP positive remains
unattributable: it may reflect dermoscopy-domain transfer, HAM10000 image-level
overlap in Derm1M, or both.

## What Was Fixed

- The paper no longer uses categorical "PMC-15M, no HAM10000" shorthand for
  BiomedCLIP. It now says PMC-15M has no documented raw HAM10000/ISIC archive
  ingestion, which is the defensible claim.
- Appendix C now matches the method section: the nuisance families share no
  operation--parameter-band combination, and JPEG is explicitly called out as
  the sole recurring transform type with non-overlapping quality bands.
- The related-work framing now says the held-out nuisance families are absent
  from the probe training distribution, not from backbone pretraining.
- `paper/Makefile` now uses local `tectonic` when available, keeps
  intermediates/logs needed for arXiv, runs quantitative-claim verification
  before `pdf` and `arxiv`, and disables macOS AppleDouble resource-fork files
  in the archive.
- `paper/verify_claims.py` and figure builders are lint-clean and formatted.
- `paper/arxiv.tar.gz` was regenerated without `._*` entries.

## Evidence That Holds Up

- `make -C paper verify`: all locked quantitative claims verified against the
  local run mirror at tolerance `0.001`.
- `make -C paper pdf`: rebuilt a 23-page PDF through Tectonic, including
  BibTeX.
- `make -C paper arxiv`: regenerated a clean source package with
  `main.tex`, `main.bbl`, section files, tables, and figures.
- `uv run ruff check`: passed.
- `uv run ruff format --check`: passed.
- `uv run pytest`: `33 passed, 2 skipped`.

## Remaining Scientific Limitations

### EXP-009 is still the main missing partition

The paper can now be published as a preliminary report, but the central positive
result remains attribution-limited. A full venue reviewer can still reasonably
ask for the non-HAM10000 dermoscopy pretraining partition before accepting a
domain-transfer claim.

### External validity is still narrow

The evidence is HAM10000-only and uses synthetic nuisance families. The paper
states this honestly. A stronger v2 should add at least one of:

- EXP-009 on non-HAM10000 dermoscopy pretraining;
- a real longitudinal / serial-imaging dataset;
- an external evaluation corpus such as PAD-UFES-20 or non-HAM10000 ISIC
  components;
- a direct probe of the learned nuisance direction, not only AUROC.

### Most negative natural-image runs are single-seed

The natural-image failures are far below random and therefore credible as
preliminary evidence, but they are not as statistically hardened as the
five-seed DermLIP and BiomedCLIP comparisons.

## Recommendation By Target

- arXiv preliminary report: yes, with the current caveat-forward framing.
- Workshop / methodology note: yes, if framed as preliminary and feedback-seeking.
- TMLR / MIDL full / MICCAI main / NeurIPS / ICLR: defer until EXP-009 or an
  equivalent attribution-closing experiment is complete.
