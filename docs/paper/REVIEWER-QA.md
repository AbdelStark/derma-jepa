# Reviewer Q&A — anticipated questions and prepared responses

**Author:** Abdelhamid Bakhta (sole)
**Companion documents:** [`docs/paper/OUTLINE.md`](OUTLINE.md), [`docs/paper/WRITING-PLAN.md`](WRITING-PLAN.md).

This document compiles anticipated reviewer objections with prepared responses. Each Q has (i) the likely paraphrase, (ii) where in the paper the response lives, (iii) a one- or two-sentence response ready to paste into a rebuttal letter or to absorb into the paper as defensive prose.

The list is organised from most likely to least likely. The first three are the questions a careful reviewer will always raise; the rest are venue- or topic-dependent.

---

## Q1. "How can the +0.66 swing be attributed to dermoscopy-domain transfer?"

**Likely framing.** "EXP-007 reaches 0.944 on a backbone that almost certainly saw HAM10000 in pretraining. The contribution claim conflates dermoscopy-domain transfer with HAM10000 image-level overlap."

**Where in the paper.** Abstract (caveat sentence), §1 (preliminary-status block), §6 (Table 1 footnote), §7.2, §7.4, §8 (entire EXP-009 section), §9 (limitation 1).

**Response.** The paper does not attribute the swing to dermoscopy-domain transfer. The abstract states verbatim that we "explicitly do not claim that this swing is dermoscopy-domain transfer." §7.4 lists the two competing hypotheses (dermoscopy-domain transfer; HAM10000 image-level overlap) and §8 reports the design and decision table for EXP-009, the partition experiment that would choose between them. Pre-empting this objection is a design goal of the paper, not a flaw to defend against.

---

## Q2. "Why was EXP-009 not run before submission? The paper would be much stronger with the partition resolved."

**Likely framing.** "Running a non-HAM10000 dermoscopy SSL pretrain is the obvious next step. Without it, the headline is unattributable. Why publish now?"

**Where in the paper.** §8.4 ("Why EXP-009 was not run before submission").

**Response.** The corpus assembly is multi-week: provenance audit on every ISIC component to confirm HAM10000 exclusion, SSL pretrain compute, pretrain-quality check on a held-out HAM10000 split. Releasing the methodology and the nine-run arc at this stage is intentional — the paper seeks community feedback on the proxy-task design and on whether the proposed partition is the right one before committing to that effort. A v2 with EXP-009 results is on the roadmap (see [`docs/paper/WRITING-PLAN.md` Phase 6](WRITING-PLAN.md)).

---

## Q3. "Why is the synthetic nuisance design meaningful for real-world dermatology?"

**Likely framing.** "The three nuisance families are deterministic operations on PIL images. Real-world dermoscopic acquisition variability looks nothing like this. Why should the result transfer?"

**Where in the paper.** §3.3 (operation lists, justified by reference to real photometric / geometric / sensor effects), §9 limitation 2.

**Response.** Each operation in the three nuisance families approximates a category of real lesion-photography acquisition variation: brightness/contrast/saturation = lighting, rotation/scale/translate/flip = framing, Gaussian blur / motion blur / posterise = camera optics and ISP, JPEG round-trips = compression, hair erasure / vignette = acquisition artefacts, gamma / color-temperature shift = ambient lighting. The proxy is synthetic by construction and §9 limitation 2 says so explicitly; the contribution is methodological — a controlled environment in which "trained on two families, evaluated on a third disjoint family" can be tested at all — not a clinical claim. Real same-lesion-over-time validation is identified as future work in §10.

---

## Q4. "How do you know the leakage probe is sufficient?"

**Likely framing.** "Patient identifiers are absent from HAM10000 metadata. How do we know the splits are leakage-controlled?"

**Where in the paper.** §3.4, Appendix G (reproducibility checklist), per-run `data_audit.json` artefacts.

**Response.** Splits are deterministic from `seed: 20260422` via [`src/derma_jepa/public_data.py:326` `_split_records`](../../src/derma_jepa/public_data.py); the splitter falls back to lesion-ID groups when patient IDs are absent, which is the case for HAM10000. Every primary-tier run produces a `data_audit.json` reporting `lesion_overlap` and `patient_overlap` zero-counts across train/val/test. The reproducibility appendix points reviewers at the audit JSON for any specific run.

---

## Q5. "The contribution is too narrow — only one dataset, only ViT-B-class architectures."

**Likely framing.** "HAM10000 is one benchmark, ViT-B is one capacity tier. The paper should generalise."

**Where in the paper.** §9 limitations 3 and 4; §8 follow-up scope.

**Response.** Acknowledged explicitly in §9. The methodology contribution (the leakage-controlled three-family-disjoint nuisance design) is independent of the empirical scope; future work extends to PAD-UFES-20 / ISIC archives and to other architectures (DINOv2 self-distillation, SigLIP, transformer-only image backbones).

---

## Q6. "Why didn't you run an MLP-on-DermLIP ablation?"

**Likely framing.** "The scaffold ablation is on DINOv2 only. What if the MLP scaffold breaks the DermLIP positive?"

**Where in the paper.** §9 limitation 6.

**Response.** §9 limitation 6 acknowledges the gap. The linear scaffold already saturates training (train AUROC 0.9999) and val/test (0.944), so an MLP on top is unlikely to move the result; this is stated as the prediction. A confirming run is straightforward to add and is the next ablation if reviewer feedback prefers it before EXP-009.

---

## Q7. "Why these specific nuisance operations? Are they cherry-picked to produce the inversion?"

**Likely framing.** "The three families look chosen to be hard. Did you tune the operations until you got the below-random result?"

**Where in the paper.** §3.3 (operation lists with parameter ranges from `_apply_*_nuisance` source), Appendix C (verbatim operation specifications).

**Response.** No tuning. The three families were specified once in `_apply_strong_nuisance`, `_apply_strong_held_out_nuisance`, and `_apply_strong_held_out_2_nuisance` in [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) and never modified after EXP-001 / EXP-003 / EXP-004 respectively. The disjointness rule (operation-level, no shared transform type) was also fixed up front. Git blame on those functions confirms zero post-EXP-004 modification of operation parameters. The deterministic-augmentation axis is a design choice, but the specific operations were not chosen to produce the inversion; the inversion was an empirical observation.

---

## Q8. "AUROC below 0.5 is the same as flipping the predictor — what does the inversion mean?"

**Likely framing.** "Test AUROC of 0.249 just means you have a working predictor with the wrong sign. Why is that interesting?"

**Where in the paper.** §7.1 (mechanism), §4.6 of every report (pair-score histograms).

**Response.** AUROC below 0.5 *would* be equivalent to a sign flip if the predictor were correctly oriented on every split. It is not: train AUROC is 0.5–1.0 in the correct direction (depending on the run); val and test AUROC are 0.25–0.30 in the inverted direction. The asymmetry is the result. A simple sign flip would invert all three; the observed pattern requires a predictor that learns family-specific directions on training and extrapolates them in the wrong sign on the unseen family. §7.1 develops this argument and the pair-score histograms in §4.6 show stable / changing distributions visibly flipping between training and test families.

---

## Q9. "BiomedCLIP saw 'figures referencing HAM10000 in research papers.' How can you claim it didn't see HAM10000?"

**Likely framing.** "PMC-15M contains figures from papers that use HAM10000. So BiomedCLIP did see HAM10000 content."

**Where in the paper.** §2.2 of EXP-008, §3.3, Appendix H.

**Response.** The PMC-15M corpus contains figure-caption pairs from open-access biomedical papers. Such papers may include figures derived from HAM10000 (e.g. rendered as low-resolution embedded illustrations), but PMC-15M was not constructed by ingesting HAM10000 or the ISIC archive. Any HAM10000 image content reaching BiomedCLIP would be downsampled, cropped, or otherwise processed for figure-rendering and would represent a tiny fraction (likely < 0.01%) of the pretraining data. Compare to DermLIP's Derm1M, which is built by collecting public dermoscopy datasets. The qualitative distinction is that BiomedCLIP saw HAM10000 content as occasional figure rendering, while DermLIP saw HAM10000 content as native dataset entries. EXP-008 §2.2 makes this distinction explicit.

---

## Q10. "Why DermLIP_ViT-B-16 and not DermLIP_PanDerm-base-w-PubMed-256? The PanDerm variant is the headline DermLIP model."

**Likely framing.** "You picked a less canonical DermLIP variant. Why?"

**Where in the paper.** §4.1 (backbone table footnote), [EXP-007 §2.1](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).

**Response.** The PanDerm variant's HF config uses an older `open_clip` schema (`pretrain_path` key) that fails to load on `open_clip 3.3.0`. Picking the OpenAI-CLIP-architecture variant gives an *exact* architecture match to EXP-006b (OpenAI CLIP ViT-B/16), so EXP-006b vs EXP-007 isolates the pretraining data alone. This is a stronger experimental design than mixing architecture and pretraining-data axes, and it is documented in EXP-007 §2.1.

---

## Q11. "Why is the predictor `weight − I` regulariser instead of standard L2 (`weight`)?"

**Likely framing.** "The regulariser shrinks toward identity, not toward zero. Standard practice is shrinkage toward zero. Justify."

**Where in the paper.** §3.2 (predictor variants), Appendix B (hyperparameters).

**Response.** The proxy task is *stable-pair latent prediction*, where the target is a small perturbation of the context image's latent. The trivial baseline is therefore $g_\theta = \mathrm{identity}$, not $g_\theta = 0$. Shrinking toward identity at initialisation places the optimiser at a known-reasonable starting point and lets gradient descent move *off* identity only insofar as the data require. Standard L2 shrinkage toward zero would push the predictor to map every input to the origin, which is the wrong inductive prior for this objective. Source: [`src/derma_jepa/training.py:242` `_fit_linear_predictor`](../../src/derma_jepa/training.py); the regulariser is `(weight − identity)` not `weight`.

---

## Q12. "How does this differ from standard distribution-shift evaluation (WILDS, in-search-of-lost-DG)?"

**Likely framing.** "Held-out evaluation under distribution shift is not new. What is new here?"

**Where in the paper.** §2 (related work, third paragraph).

**Response.** WILDS-style covariate-shift benchmarks vary the *natural* distribution between train and test (different hospitals, different scanners, different countries). The contribution here is a *deterministic* nuisance-family decomposition that lets the experimenter precisely control what the test family is and is not. The disjointness rule is operation-level, not parameter-level, which means the nuisance applied at test time was *categorically* unseen at training time. WILDS does not isolate this axis; the contribution complements rather than replaces WILDS-style evaluations.

---

## Q13. "Why hasn't a direction-structure probe (PCA of stable-pair difference vectors per family) been run?"

**Likely framing.** "§7.2 hypothesises nuisance-direction alignment; this is testable. Why not test it?"

**Where in the paper.** §7.2, §10 (open issues), §8 follow-up.

**Response.** The probe is compute-cheap and is on the v0.1-or-v0.2 to-do list (see [`docs/paper/OUTLINE.md` §10](OUTLINE.md)). If the writing-pass schedule allows, it will land in v1.0 and §7.2's hypothesis will become a result.

---

## Q14. "What about the third experiment (EXP-001) reaching ceiling? That isn't really a result."

**Likely framing.** "EXP-001 just shows the proxy was too easy. Why include it?"

**Where in the paper.** §5.1.

**Response.** EXP-001 is reported because it justifies the hardening that produces EXP-002 and onward. Without it the methodology section's "trivial proxy → hardened proxy → held-out proxy → mixed-family-third-held-out proxy" progression would lack a starting point, and the iterative nature of the proxy design (the contribution) would be opaque. EXP-001 is one paragraph in §5; cost is minimal, value is a complete methodology trajectory.

---

## Q15. "Why use AUROC and not a bespoke metric for change detection?"

**Likely framing.** "A change-detection task should use detection-theoretic metrics, not AUROC."

**Where in the paper.** §3.5.

**Response.** Every run reports AUROC, AUPRC, equal-error-rate threshold, and FPR at fixed TPR = 0.8 — the standard suite for binary detection-theoretic evaluation. AUROC is the headline because it is threshold-independent and most directly comparable across baselines and runs; the other three are reported for completeness. Source: [`src/derma_jepa/metrics.py:29` `binary_metric_summary`](../../src/derma_jepa/metrics.py).

---

## How to use this document

- Before submission: read every Q and confirm the cited paper section actually contains the response. Update either the paper or this document if they disagree.
- During rebuttal: paste the response sections verbatim into the rebuttal letter, prefixed by the reviewer's question.
- After rebuttal: append any new questions raised during review with their responses, so v2 of the paper can pre-empt them.

---

## Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-28 | Abdelhamid Bakhta | Initial Q&A. 15 anticipated questions with prepared responses, organised by likelihood. |
