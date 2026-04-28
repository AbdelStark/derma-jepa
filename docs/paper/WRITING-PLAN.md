# DermaJEPA paper — writing plan

**Author:** Abdelhamid Bakhta (sole)
**Companion:** [`docs/paper/OUTLINE.md`](OUTLINE.md) — section-by-section content and source pointers.
**Goal:** A preliminary preprint released on arXiv, then a TMLR / ML4H / MIDL-short submission, with v2 contingent on EXP-009.

This document is the operational checklist for taking the outline to a submitted PDF. It assumes the OUTLINE is the source of truth for what the paper says; this plan is the source of truth for how it gets written.

---

## Phase 0 — repository setup (target: half a day)

- [ ] Create a `paper/` directory at the top of the repo (sibling to `src/`, `docs/`, etc.) for LaTeX sources and figure assets, separate from `docs/paper/` which is project-management content. Final structure:
  ```
  paper/
    main.tex
    refs.bib
    sections/
      00-abstract.tex
      01-introduction.tex
      02-related-work.tex
      03-method.tex
      04-experimental-setup.tex
      05-experimental-sequence.tex
      06-results.tex
      07-analysis.tex
      08-followup-exp009.tex
      09-limitations.tex
      10-conclusion.tex
      A-per-run.tex
      ...
    figures/
      fig1-cross-backbone-gradient.pdf
      fig2-train-vs-test-scatter.pdf
      fig3-pair-score-histograms.pdf
      method-diagram.pdf
      experimental-design.pdf
      pipeline.pdf
    tables/
      tab1-cross-run.tex
      tab2-predictor-optimiser.tex
      tab3-seed-sweep.tex
      tab4-rep-health.tex
    Makefile
  ```
- [ ] Pick a LaTeX template. For arXiv: a clean `article`-class template. For TMLR: the official TMLR class. For MIDL / MICCAI / ML4H: their respective templates. Recommendation: start with `article` for v0.1 arXiv submission, keep section files venue-agnostic so the wrapper can be swapped.
- [ ] Add a `Makefile` with `make pdf`, `make figures`, `make tables`, `make clean` targets.
- [ ] Wire BibTeX entries for every reference in `refs.bib` (see [§5 of this plan](#phase-5--references-and-bibliography)).
- [ ] First commit: empty section files with the correct section headings and `\todo{}` placeholders. Use the `todonotes` package so `\listoftodos` shows progress.

---

## Phase 1 — figures and tables (target: 2 days)

The figures are the spine of the paper. Build them before the prose.

### Figure 1 — cross-backbone gradient

- Data: cross-run table values from [`docs/paper/OUTLINE.md` §5](OUTLINE.md) ("Numbers safe to cite").
- Tool: `matplotlib`. Horizontal bar plot, four bars (DINOv2 0.249, OpenAI CLIP 0.286, BiomedCLIP 0.329, DermLIP 0.944). Pixel-L2 reference line at 0.580.
- Annotations: error bars from bootstrap CIs / seed std. DermLIP bar carries a "[contamination caveat]" label to the right.
- Build: write `paper/figures/build_fig1.py` reading the locked numbers from a small JSON fixture committed alongside.

### Figure 2 — train-vs-test AUROC scatter

- Data: per-run `metrics.json` from `outputs/hf-runs/<run-id>/` for runs 4–8. Use the seed-mean values for EXP-007 / EXP-008.
- Tool: `matplotlib`. X = train AUROC, Y = test AUROC. Each point labelled with EXP ID. Diagonal `y = x` reference line. Horizontal random-baseline line at 0.5.
- Build: `paper/figures/build_fig2.py` consuming the full metrics JSONs already pulled to `outputs/hf-runs/`.

### Figure 3 — three-panel pair-score histograms

- Data: `outputs/hf-runs/<run-id>/artifacts/embeddings/jepa_predictor_latents.npz` (`score`, `label`, `split`) for `ham10000-hf-clip-exp006b-v1`, `ham10000-hf-dermlip-exp007-v1`, `ham10000-hf-biomedclip-exp008-v1`. Filter to `split == "test"`.
- Tool: `matplotlib` `subplots(1, 3, sharex=True, sharey=True)`; histograms of `score` per `label`.
- Build: `paper/figures/build_fig3.py`. Use the same bin grid across panels for visual comparability.

### Tables

- Build all tables as standalone `.tex` files using `booktabs` and `siunitx` for number alignment.
- Table 1 (cross-run): hand-write from the OUTLINE numbers; the data is small and stable.
- Table 2 (predictor × optimiser): hand-write from EXP-004 / 005 / 006a §4.1.
- Table 3 (seed sweep): hand-write from the seed-sweep summary §3.
- Table 4 (representation health, appendix): script-generate by reading `representation_health` from each `metrics.json`.

### Diagrams

- The README Mermaid sources for predictor objective, experimental design, and pipeline are good first-cut content. For the paper, render to vector PDF.
- Two options: (a) Render Mermaid → SVG via `mmdc`, then SVG → PDF via Inkscape. (b) Re-author in TikZ. Option (a) is faster; option (b) is camera-ready quality.
- Recommendation: option (a) for arXiv v1; option (b) before any conference submission.

---

## Phase 2 — drafting order (target: 5 working days for a complete v0.1)

Recommended order, easiest-to-hardest, also approximating bottom-up dependencies:

1. **Section §3 Method.** Most code-grounded; the writing task per subsection is mostly translation from `src/derma_jepa/`. Day 1.
2. **Section §4 Experimental setup.** Backbones table is already in the README; lift and adapt. Day 1.
3. **Section §5 Experimental sequence.** Nine paragraphs at ~80 words each, each compressed from one EXP report's §1 Summary. Day 2.
4. **Section §6 Results.** Tables and figures already exist by this point; this section is captioning + 1–2 paragraphs of prose. Day 3 morning.
5. **Section §9 Limitations.** Numbered list, ~300 words total. Day 3 afternoon.
6. **Section §8 Required follow-up — EXP-009.** Decision table + prose. Day 3 afternoon.
7. **Section §7 Analysis and discussion.** Hardest because it's the highest-judgement section. Day 4.
8. **Section §10 Conclusion.** Short. Day 4 afternoon.
9. **Section §2 Related work.** Requires re-reading 6–8 papers. Day 4 evening.
10. **Section §1 Introduction.** Written last so it can summarise an already-drafted paper. Day 5 morning.
11. **Abstract.** Written after §1. Day 5 morning.
12. **Title.** Pick from OUTLINE §1 after the abstract is done.

For each section, the writing loop is:

1. Open the corresponding subsection in `OUTLINE.md`.
2. Read the source pointers — confirm the claims still hold, copy headline numbers from the "Numbers safe to cite" block.
3. Draft into the section file. Add `\citep{}` placeholders against `refs.bib`.
4. Run `make pdf`; check the section renders without overfull boxes.
5. Tick the writing-task checkboxes in the OUTLINE subsection.

---

## Phase 3 — internal review passes

After v0.1 is complete:

### Pass A — claim audit (1 day)

- For every numerical claim in the paper, find it in OUTLINE §5 ("Numbers safe to cite"). If it isn't there, either add it (with a source pointer) or rewrite the claim.
- For every "we proved / we observed / we showed" sentence, check that it doesn't overstate. Specifically:
  - "Solves the proxy" → not allowed for DermLIP.
  - "Domain-pretraining unlocks JEPA" → not allowed without the contamination caveat in the same sentence.
  - "BiomedCLIP rules out" → fine if scoped to "any medical-image pretraining as sufficient."
- For every figure caption, check that it states the data source and the run IDs.

### Pass B — reproducibility audit (0.5 day)

- For every experiment cited in the body, the corresponding `docs/experiments/EXP-NNN-…-v1.md` exists, the run ID is on the run archive, and `derma-jepa hf-run summary --run-id <id>` produces the cited numbers.
- For every figure / table, the build script exists in `paper/figures/` or `paper/tables/`, runs cleanly, and produces the committed asset.
- Reproducibility checklist appendix (Appendix G in OUTLINE §7) — fill in every item.

### Pass C — humanise the prose (0.5 day)

- Strip emphatic adjectives ("decisively", "strikingly", "remarkably", "notably", "crucially").
- Strip rule-of-three padding ("Three reads:", "Two consequences:").
- Strip narrative framing ("the arc", "Act I/II/III", "the load-bearing axis", "the binding constraint").
- Replace italics for emphasis with grammatical italics only.
- Trim every sentence that says "in this paper we" / "we now turn to" / similar transitions.
- Read every paragraph out loud. If it sounds promotional, rewrite.

### Pass D — adversarial read (0.5 day)

- Read the paper as a hostile reviewer. Specific reviewer questions to anticipate:
  1. *Contamination.* "How can the +0.66 swing be attributed to dermoscopy-domain transfer?" → Answer: it cannot, the paper says so, EXP-009 is the experiment.
  2. *Synthetic nuisance.* "Why is this evaluation meaningful for real-world dermatology?" → Answer: §9 limitation 2; the proxy is synthetic by design and is a methodology contribution, not a clinical claim.
  3. *Single architecture.* "Why only ViT-B/16-class backbones?" → Answer: §9 limitation 3; one-axis-at-a-time methodology required holding architecture constant; future work.
  4. *Single dataset.* "Why only HAM10000?" → Answer: §9 limitation 4; HAM10000 has the lesion-ID metadata required for the leakage-controlled design; PAD-UFES-20 is a candidate for v2.
  5. *Why not run EXP-009 first.* "This paper would be much stronger with the partition resolved." → Answer: §8.4; the paper is explicit about being a preliminary release seeking feedback before committing to multi-week corpus assembly.

- For each anticipated objection, ensure the paper has an explicit sentence that handles it.

---

## Phase 4 — final renders and pre-flight (target: 1 day)

- [ ] Final figure render at 300 DPI, vector where possible.
- [ ] Final table compile under `booktabs` with `siunitx` for number alignment.
- [ ] Spell-check pass via `aspell` or similar.
- [ ] arXiv compatibility: compile under `pdflatex` (no `xelatex`-only features unless required); ensure `arxiv.tar.gz` builds with the source tree as-is.
- [ ] PDF size check (< 10 MB for arXiv ease).
- [ ] Final BibTeX consistency pass: every `\cite{}` resolves; every `refs.bib` entry has a DOI or arXiv ID.

---

## Phase 5 — references and bibliography

`paper/refs.bib` should include, at minimum:

- I-JEPA — `@inproceedings{assran2023ijepa, ... arxiv = {2301.08243}}`
- JEPA position paper — LeCun 2022 OpenReview ID.
- DINO — `@inproceedings{caron2021dino, ... arxiv = {2104.14294}}`
- DINOv2 — `@article{oquab2024dinov2, ... arxiv = {2304.07193}}`
- CLIP — `@inproceedings{radford2021clip, ... arxiv = {2103.00020}}`
- OpenCLIP — `@inproceedings{cherti2023openclip, ... arxiv = {2212.07143}}`
- HAM10000 — `@article{tschandl2018ham10000, doi = {10.1038/sdata.2018.161}}`
- DermLIP / Derm1M — `@misc{yan2025derm1m, ... arxiv = {2503.14911}}`
- BiomedCLIP — `@misc{zhang2023biomedclip, ... arxiv = {2303.00915}}`
- MAE — `@inproceedings{he2022mae, ... arxiv = {2111.06377}}`
- VICReg — `@inproceedings{bardes2022vicreg, ... arxiv = {2105.04906}}`
- WILDS — `@inproceedings{koh2021wilds, ... arxiv = {2012.07421}}`
- DG search — `@inproceedings{gulrajani2021dg, ... arxiv = {2007.01434}}`
- BCN20000 — `@misc{combalia2019bcn20000, ... arxiv = {1908.02288}}`
- Foundation models — `@misc{bommasani2021foundation, ... arxiv = {2108.07258}}`
- ISIC challenge papers (2017, 2018, 2019) — three entries.

Each `bib` entry should have at minimum: `author`, `title`, `year`, and one of `arxiv` / `doi` / `url`. Run `bibtex` warnings to zero before submission.

---

## Phase 6 — submission

### v0.1 — arXiv preprint

- [ ] Final read by author.
- [ ] Tag the source repo at the matching commit, e.g. `git tag paper-v0.1`.
- [ ] Create `arxiv.tar.gz` with `make arxiv`.
- [ ] Submit to arXiv `cs.LG` primary, with `cs.CV` and `eess.IV` as cross-listings.
- [ ] After acceptance: add the arXiv ID and DOI back to `README.md` (replace the BibTeX placeholder), the run-archive HF dataset card, and `docs/paper/OUTLINE.md`.

### v1.0 — venue submission

After arXiv preprint is live and (if applicable) feedback has been incorporated:

- TMLR: rolling submission. Re-format under TMLR class. Submit; respond to reviewer rounds; final-decision typically 6–10 weeks.
- ML4H 2026: workshop, NeurIPS-aligned. Re-format under workshop class.
- MIDL 2026 short: 4 pp + refs format. Trim aggressively.

For each venue, keep the section files in `paper/sections/` venue-agnostic; only the wrapper (`main.tex`) and bibliography style change.

### v2.0 — post-EXP-009

If EXP-009 gets run:

- Add Appendix I (EXP-009 results) and incorporate the partition outcome into §1, §5, §7, §10.
- Update the contributions list and the headline based on the EXP-009 decision-table band.
- Revisit the title; if EXP-009 confirms ≥ 0.85, the "preliminary" qualifier can drop.

---

## Phase 7 — what to ship alongside the paper

- The repository at the tagged commit, MIT-licensed.
- The HF run-archive dataset, public.
- A "supplementary materials" zip that bundles the figure-build scripts, the table source files, and a reproducibility README pointing back to this plan.
- A short blog post or HF Space (optional, post-arXiv) summarising the headline figure and the contamination caveat in lay terms — strict scope-of-claim, no clinical framing.

---

## Open decisions

- Whether to run a direction-structure probe (PCA of stable-pair difference vectors per family for DermLIP vs OpenAI CLIP) before submission. Cheap, would let §7.2 become a result rather than a hypothesis. Recommendation: yes, if it can be done in a day.
- Whether to add a non-HAM10000 dataset (e.g. PAD-UFES-20) as an evaluation-only secondary split for the v1 paper. Useful but adds scope; defer to v2 unless reviewer feedback demands it.
- Whether to swap the BiomedCLIP partition for a stricter one (e.g. MedSigLIP-448). Recommendation: defer; one general-medical point is enough for the v1 paper, and MedSigLIP's HAI-DEF terms add friction.

---

## Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-28 | Abdelhamid Bakhta | Initial writing plan. |
