# DermaJEPA — research notes

## Project domain

Dermatology imaging and longitudinal change detection.

## Central research question

A JEPA-style encoder can learn a lesion representation where stable lesions remain close under nuisance variation and meaningful lesion evolution appears as a structured latent trajectory departure.

## Data sources under consideration

- ISIC Archive
- HAM10000
- PAD-UFES-20
- Optional same-lesion subset curation if enough repeated examples exist

## Questions to answer before implementation grows

1. What is the smallest task framing that still tests the thesis honestly?
2. What is the strongest cheap baseline?
3. What nuisance or confounders matter most in practice?
4. What would count as a fake win here?
5. What result would actually persuade a skeptical technical reader?

## Findings after EXP-001 → EXP-008 + seed sweep

The questions above motivated the experimental sequence in `docs/experiments/`. Provisional answers from nine primary-tier runs on HAM10000:

1. **Smallest honest task framing.** A leakage-controlled stable-vs-changing pair task on HAM10000 with three disjoint synthetic nuisance families (`strong`, `strong_held_out`, `strong_held_out_2`), training on a mixture of two and evaluating on the third unseen family, separates "fits the seen distribution" from "generalises out of distribution" cleanly. The matched-eval version (EXP-002) is too easy in the sense that any reasonable scaffold + frozen backbone wins; the third-family-held-out version (EXP-004 onward) is hard enough that frozen natural-image backbones invert below random.
2. **Strongest cheap baseline.** Pixel L2 (AUROC 0.580 on `strong_held_out_2`). It barely exceeds random, but it exceeds every frozen-natural-image-backbone scaffold tested under nuisance-held-out evaluation, and it exceeds frozen general-medical-pretrained BiomedCLIP. SSIM and frozen-cosine baselines do worse and frequently invert.
3. **Nuisance directions that matter.** The `strong_held_out_2` family — a third disjoint synthetic-nuisance configuration unseen at training time — is the binding constraint on whether a frozen-backbone JEPA scaffold generalises. Under DINOv2 and OpenAI CLIP, the linear predictor learns directions specific to the seen training families and extrapolates them in the wrong way to the third family, producing below-random test AUROC. Under DermLIP, the same scaffold learns a direction that transfers to the third family.
4. **What a fake win looks like.** Two specific fake-win patterns surfaced. First, EXP-002's matched-eval delta of +0.27 AUROC was real on its proxy but did not transfer to a held-out nuisance family (EXP-003 dropped that delta to −0.28). Second, EXP-007's headline +0.36 AUROC over the strongest baseline depends on a backbone (DermLIP) whose pretraining corpus almost certainly includes HAM10000; without partitioning dermoscopy-domain transfer from HAM10000 image-level overlap (EXP-009), the headline cannot be claimed as out-of-distribution generalisation in the strict sense.
5. **What persuades a skeptical reader.** The nine-run sequence with explicit train-and-eval-on-disjoint-nuisance-families, identical scaffolds varied only along one axis at a time (predictor class, optimiser, backbone, pretraining-data domain), seed sweeps locking the two headline numbers, and an honest contamination caveat front-loaded. Each report cites the run's Hub location and a single `derma-jepa hf-run summary` command reproduces the headline numbers.

The current open question, scoped as EXP-009, is whether the EXP-007 win comes from dermoscopy-domain transfer in the non-HAM10000 portion of Derm1M or from HAM10000 image-level overlap during DermLIP's pretraining.

## Research anti-patterns

- Overclaiming beyond the chosen task.
- Broadening the problem until the evaluation becomes vague.
- Substituting visual polish for empirical clarity.
- Comparing only against weak baselines.
- Letting the demo narrative outrun the measured result.

## Research spikes

- `docs/research-spikes/2026-04-21-ml-intern-and-medical-sam.md` — evaluation of Hugging Face `ml-intern` and whether a SAM-style medical segmentation workflow should be used as a DermaJEPA sidecar.
