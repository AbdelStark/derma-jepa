# DermaJEPA paper outline (v2, preliminary release, deepened)

**Author:** Abdelhamid Bakhta (sole)
**Status:** Working outline for a preliminary preprint. The paper releases the nine-experiment arc, the proxy-task design, the run archive, and the locked seed-sweep numbers as a methodology-and-early-evidence contribution. EXP-009 (the dermoscopy-domain-transfer vs HAM10000-contamination partition) is **not** in scope for this paper; it is reported as required follow-up work with an explicit decision table.
**Source repository:** [github.com/AbdelStark/derma-jepa](https://github.com/AbdelStark/derma-jepa)
**Run archive:** [`abdelstark/derma-jepa-runs`](https://huggingface.co/datasets/abdelstark/derma-jepa-runs)
**Companion document:** [`docs/paper/WRITING-PLAN.md`](WRITING-PLAN.md) — operational checklist for taking this outline to a submitted PDF.

---

## 0. Framing

This paper is a *preliminary report*, not a closed result. Sharing the arc at this stage is justified by:

1. The methodology — a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families and a `train-on-two / evaluate-on-third` protocol — is novel enough that it should be put in front of reviewers before further compute is spent extending it.
2. The nine-run sequence already characterises a clean failure mode of frozen natural-image and frozen general-medical backbones (test AUROC inverts below random across two architectures, three predictor scaffolds, two optimisers).
3. The single positive result observed (frozen DermLIP at 0.944 ± 0.003 across 5 seeds) **cannot be attributed to dermoscopy-domain transfer with the experiments in this paper alone**. DermLIP's pretraining corpus, Derm1M, almost certainly contains HAM10000 raw images, and the BiomedCLIP partition (EXP-008) only rules out the broader "any medical-image pretraining" alternative — not HAM10000 image-level overlap specifically.
4. EXP-009 (a self-pretrained DINOv2 on a non-HAM10000 dermoscopy corpus) is the experiment that would partition dermoscopy-domain transfer from contamination. Building that corpus and running the SSL pretrain is a multi-week engineering task. Releasing the methodology at this stage to validate the proxy design and gather feedback on the partition before committing to it.

**Source pointers for the framing claims:**

- Three-family disjointness: [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) (manifest builder, nuisance-family wiring) and [`docs/spec/MVP-SPEC.md` §8](../spec/MVP-SPEC.md) (nuisance augmentation suite contract).
- Train-on-two / evaluate-on-third: [`configs/data/ham10000_hf_mounted_exp004.yaml`](../../configs/data/ham10000_hf_mounted_exp004.yaml) — the canonical proxy config used by every run from EXP-004 onward.
- Below-random inversion claim: [`docs/experiments/EXP-004-…-v1.md` §4](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md), [`docs/experiments/EXP-005-…-v1.md` §4](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md), [`docs/experiments/EXP-006a-…-v1.md` §4](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md), [`docs/experiments/EXP-006b-…-v1.md` §4](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md).
- DermLIP positive + caveat: [`docs/experiments/EXP-007-…-v1.md` §1, §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).
- BiomedCLIP partition: [`docs/experiments/EXP-008-…-v1.md` §1, §4.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).
- Seed sweep: [`docs/experiments/EXP-007-008-seed-sweep-summary.md`](../experiments/EXP-007-008-seed-sweep-summary.md).

---

## 1. Title

Three working titles, all explicit about the preliminary status:

1. **Frozen-backbone JEPA-style probes on HAM10000: preliminary results from a nine-experiment ablation, with an unpartitioned contamination caveat.**
2. *Toward a clean test of frozen-backbone JEPA generalisation on dermoscopy: methodology, early evidence, and an open partition experiment.*
3. *When does frozen-backbone JEPA generalise on dermoscopy? A preliminary nine-experiment study and a planned partition.*

Final title locks at submission. The "preliminary" qualifier stays in the title regardless.

**Writing task.** Pick option 1 unless reviewer feedback objects. Option 2's "Toward" framing is honest but can read as hedging.

---

## 2. Abstract (≈260 words)

Self-supervised representations promise robustness to nuisance variation; whether that robustness transfers across novel nuisance families is rarely tested. We probe this on a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families — partitioned along the deterministic-augmentation axis — training a JEPA-style latent predictor on stable pairs from two of three families and evaluating on the third unseen family. We report a preliminary nine-experiment arc. Across two frozen natural-image backbones (DINOv2 ViT-B/14, OpenAI CLIP ViT-B/16), three predictor scaffolds (linear, underfit MLP, fit MLP under Adam), and two optimisers (SGD, Adam), test AUROC on the held-out family is 0.25–0.29 — below random and below cheap baselines (pixel L2, SSIM, raw embedding cosine). A frozen general-medical backbone (BiomedCLIP, PMC-15M, no HAM10000) lifts test AUROC by only +0.04 to 0.329 ± 0.012 across 5 seeds. A frozen dermoscopy-pretrained backbone (DermLIP, Derm1M) lifts test AUROC to 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline and the only above-random result on the held-out family across nine runs. Holding architecture (ViT-B/16) and predictor (linear) constant, varying only the pretraining-data domain produces a +0.66 AUROC swing localised to the dermoscopy-specific step. **The paper explicitly does not claim that this swing is dermoscopy-domain transfer**: DermLIP's Derm1M corpus almost certainly contains HAM10000, and the experiments reported here cannot separate dermoscopy-domain transfer from HAM10000 image-level overlap. The methodology, configurations, and run archive are released, and a planned partition experiment (a non-HAM10000 dermoscopy SSL pretrain) is presented as required follow-up. The contribution is the proxy-task design, the failure characterisation, and a clearly scoped open question — not a conclusion about whether frozen-backbone JEPA generalises on dermoscopy.

**Writing task.** Final pass before submission: replace any first-person plural ("we report", "we release") with passive voice if a sole-author paper for the venue requires it; otherwise keep "we" since most ML venues accept it for sole-author papers as the editorial / authorial voice.

---

## 3. Section structure

Each subsection lists (i) what the section says, (ii) source pointers for each claim, (iii) writing tasks.

### §1. Introduction (≈1.0 page)

Content:

- **Motivation.** Lesion monitoring is fundamentally a change-detection problem under heavy nuisance variation (illumination, angle, hair, framing). Self-supervised representations are pitched as nuisance-invariant; the question is whether that invariance generalises across novel nuisance families a model never saw during training.
- **Setting.** Public HAM10000 dermoscopy data is cross-sectional, so a leakage-controlled longitudinal proxy is constructed: lesion-ID-aware splits, synthetic stable-pair augmentations applied post-split, changing-pair construction matched on diagnosis and anatomical site.
- **Probe.** A JEPA-style latent predictor (linear or 2-layer identity-residual MLP) over a frozen vision backbone, trained to minimise L2 between predicted and observed target latents on stable pairs.
- **One-axis-at-a-time ablation.** Across nine experiments, predictor class, optimiser, and backbone are varied independently while every other knob is held constant.
- **Preliminary status.** Explicit paragraph: this is an early-stage report; the EXP-007 positive result is unpartitioned with respect to HAM10000 contamination; EXP-009 is required to close the question; releasing now to validate the methodology and to invite feedback on the partition design and on what a stricter test would look like.
- **Contributions** (mirrored in §10).
- **Headline figure.** Figure 1: cross-backbone gradient.

Source pointers:

- "Lesion monitoring as change-detection": [`docs/prd/PRD.md` §3](../prd/PRD.md), [`docs/spec/MVP-SPEC.md` §1](../spec/MVP-SPEC.md), [`docs/rfcs/RFC-0001-problem-statement-and-product-thesis.md`](../rfcs/RFC-0001-problem-statement-and-product-thesis.md).
- "HAM10000 is cross-sectional": [Tschandl et al., *Sci. Data* 2018](https://doi.org/10.1038/sdata.2018.161); see also [`data/README.md`](../../data/README.md) (metadata fields, no patient-ID timestamps).
- Synthetic stable-pair augmentations: [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) — `apply_nuisance_*` functions; [`docs/rfcs/RFC-0002-data-corpus-and-longitudinal-proxy-design.md`](../rfcs/RFC-0002-data-corpus-and-longitudinal-proxy-design.md).
- Lesion-ID splits: [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) — `_split_groups` / `_assign_split`; tests in [`tests/test_public_data_pipeline.py`](../../tests/test_public_data_pipeline.py).
- Predictor objective: [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) — `train_jepa_predictor`, the L2-MSE loop and identity warm-start.
- Per-experiment configs: [`configs/data/ham10000_hf_mounted_exp00*.yaml`](../../configs/data/).
- Per-experiment launchers: [`scripts/hf_jobs_ham10000_exp00*.sh`](../../scripts/).

Writing tasks:

1. Draft motivation paragraph (~120 words) that does not promise clinical utility.
2. Draft setting paragraph (~80 words) that names HAM10000 as cross-sectional and the proxy as synthetic.
3. Draft probe paragraph (~100 words) that defines JEPA-style narrowly (frozen backbone, single-step, stable-pair) and cites I-JEPA and the JEPA position paper.
4. Draft preliminary-status paragraph (~80 words) — the contamination caveat is named, EXP-009 is named, the feedback ask is explicit.
5. Insert Figure 1 reference and a one-sentence reading instruction.

### §2. Related work (≈0.75 page)

Content (one paragraph each):

- **JEPA-style and self-supervised image representations.** Position the probe as "the frozen-backbone, single-step, stable-pair subset of the JEPA family applied as an evaluation probe rather than a pretraining objective." Cite I-JEPA, JEPA position paper, DINO, DINOv2, CLIP, OpenCLIP.
- **Domain-specific medical foundation models.** BiomedCLIP, DermLIP / PanDerm, MedCLIP, MONET. State which weights are used and which are referenced for context.
- **Distribution-shift evaluation in dermoscopy / medical imaging.** ISIC challenge series, RETFound and similar foundation-model evaluations, WILDS-style covariate-shift benchmarks, in-search-of-lost-DG. Position the contribution as the deterministic, leakage-controlled, three-family-disjoint nuisance design.

Source pointers (each referenced backbone in §2 must point at the loader):

- DINOv2 loader: [`src/derma_jepa/embeddings.py`](../../src/derma_jepa/embeddings.py) `_dinov2_matrix`.
- OpenAI CLIP loader: [`src/derma_jepa/embeddings.py`](../../src/derma_jepa/embeddings.py) `_clip_matrix`.
- DermLIP / BiomedCLIP loader: [`src/derma_jepa/embeddings.py`](../../src/derma_jepa/embeddings.py) `_open_clip_matrix`.
- Backbones table: [`README.md` §Backbones evaluated](../../README.md).

Writing tasks:

1. Read the I-JEPA paper [arXiv:2301.08243], the JEPA position paper, and one DINOv2 follow-up to refresh framing.
2. Read the BiomedCLIP paper [arXiv:2303.00915] §3 (PMC-15M description) for the contamination claim about "no HAM10000 in pretraining."
3. Read the DermLIP / Derm1M paper [arXiv:2503.14911] §3 to confirm that the per-source breakdown is not published; cite the absence as the basis for "almost certainly includes HAM10000."
4. Skim WILDS [Koh et al. 2021] and Gulrajani-Lopez-Paz [2021] for the framing of held-out-family evaluation, and cite both.

### §3. Method (≈1.0 page)

Content:

- **§3.1 Notation and probe.** Frozen $f$, normalised $z = f(x) / \lVert f(x) \rVert_2$, stable / changing pair definitions, predictor $g_\theta : \mathbb{R}^d \to \mathbb{R}^d$, L2-MSE training loss, cosine-distance evaluation score.
- **§3.2 Predictor variants.** Linear with identity warm-start; 2-layer identity-residual MLP with zero-init `W_2`.
- **§3.3 Three nuisance families.** Operation lists for `strong`, `strong_held_out`, `strong_held_out_2`. Disjointness rule.
- **§3.4 Splits and pair construction.** Lesion-ID-aware splits (5,229 / 1,120 / 1,121); 1,000 stable + 1,000 changing pairs per split; strict same-diagnosis-site changing-pair policy.
- **§3.5 Evaluation protocol.** AUROC primary; 1,000-sample bootstrap 95 % CIs; AUPRC; EER threshold; FPR-at-fixed-TPR(0.8); three representation-health checks (prediction-norm, dimension-variance, collapse flag).

Source pointers:

- Predictor architecture (linear): [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) — `LinearPredictor` (or the equivalent linear branch) with identity warm-start and `weight − I` regulariser.
- Predictor architecture (MLP): [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) — `MLPPredictor` with the zero-init residual.
- L2-MSE objective and SGD/Adam dispatch: [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py).
- Cosine-distance scoring at eval: [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) and [`src/derma_jepa/baselines.py`](../../src/derma_jepa/baselines.py) for the cosine-baseline equivalent.
- Three nuisance families (operation lists): [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) — `_apply_strong_nuisance`, `_apply_strong_held_out_nuisance`, `_apply_strong_held_out_2_nuisance` (or the canonical names in the file).
- Lesion-ID-aware split assignment: [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) — `_split_groups`, deterministic seeding by `seed: 20260422`.
- Pair construction (strict same-diagnosis-site changing pairs): [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) — `_match_changing_pair`.
- Bootstrap CI implementation: [`src/derma_jepa/metrics.py`](../../src/derma_jepa/metrics.py) — `_bootstrap_auroc`, `compute_metrics`. Tests: [`tests/test_metrics.py`](../../tests/test_metrics.py).
- Representation health: [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) — `_collapse_checks` / equivalent. Per-run §4.5 in every report.
- Canonical proxy config: [`configs/data/ham10000_hf_mounted_exp004.yaml`](../../configs/data/ham10000_hf_mounted_exp004.yaml).

Writing tasks:

1. Read [`src/derma_jepa/training.py`](../../src/derma_jepa/training.py) end-to-end to confirm the linear and MLP variants match the description; rewrite §3.2 if anything has drifted.
2. Read [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py) §nuisance to confirm the disjointness claim and to extract the operation tables for §3.3.
3. Render the predictor diagram (Mermaid → TikZ); see [`README.md` § Method](../../README.md) for the source.
4. Render the experimental-design diagram (Mermaid → TikZ); see [`README.md` § Experimental design](../../README.md).
5. Cite the bootstrap-CI protocol via [`tests/test_metrics.py`](../../tests/test_metrics.py) for reviewer reproducibility.

### §4. Experimental setup (≈0.5 page)

Content:

- **§4.1 Backbones.** Table of five backbones (DINOv2 S/14, DINOv2 B/14, OpenAI CLIP B/16, DermLIP B/16, BiomedCLIP B/16) with HF Hub paths and primary citations.
- **§4.2 Training.** SGD vs Adam dispatch, 200 epochs at batch 128, identity warm-start.
- **§4.3 Compute.** Single A10G 24 GB on Hugging Face Jobs; ~85 min wall-clock per run end-to-end.
- **§4.4 Code, configurations, and run archive.** GitHub repo + HF dataset link.

Source pointers:

- Backbone table source-of-truth: [`README.md` § Backbones evaluated](../../README.md).
- Training config schema: [`src/derma_jepa/config.py`](../../src/derma_jepa/config.py) — `TrainingConfig`, `EmbeddingModelConfig`.
- Compute pins: [`scripts/hf_jobs_constraints.txt`](../../scripts/hf_jobs_constraints.txt).
- Per-run wall-clock breakdowns: §3 of every experiment report under [`docs/experiments/`](../experiments/README.md).

Writing tasks:

1. Replicate the Backbones table from the README into the paper, then add the embedding dim column.
2. Confirm the constraint file reflects the actual versions used in the EXP-007/008 runs (read the file).
3. Pull the per-run wall-clock numbers and produce a single "compute envelope" sentence.

### §5. Experimental sequence (≈1.0 page)

Content (one paragraph per stage):

- **§5.1 Sanity (EXP-001).** Trivial proxy: baselines at AUROC ≈ 1.0. Proxy needs hardening.
- **§5.2 Hardened proxy, matched eval (EXP-002).** JEPA wins +0.27 AUROC over the strongest baseline (DINOv2-S cosine = 0.652).
- **§5.3 One-family-held-out (EXP-003).** Win collapses to a loss (−0.28 vs SSIM = 0.961).
- **§5.4 Three-family training, third-family eval (EXP-004).** Predictor inverts to 0.249 AUROC, −0.33 below pixel L2 = 0.580.
- **§5.5 Scaffold ablation (EXP-005, EXP-006a).** MLP under SGD underfits (train 0.572, test 0.270); MLP under Adam fits (train 0.893) and lands at the same test 0.248. Scaffold-capacity hypothesis falsified.
- **§5.6 Backbone ablation, natural-image (EXP-006b).** OpenAI CLIP ViT-B/16 lands at test 0.286 — within bootstrap noise of DINOv2's 0.249.
- **§5.7 Pretraining-domain ablation (EXP-007).** DermLIP lands at test 0.945. Train-test drop collapses from −0.70 to −0.05. Contamination caveat introduced.
- **§5.8 Pretraining-domain partition (EXP-008).** BiomedCLIP at 0.325. Web → general-medical step is +0.04; general-medical → dermoscopy step is +0.62.
- **§5.9 Seed robustness.** 5-seed sweep on EXP-007 (0.944 ± 0.003) and EXP-008 (0.329 ± 0.012).

Source pointers (one row per EXP report; cite only by section number, never by line number):

- EXP-001 — [report](../experiments/EXP-001-ham10000-jepa-primary-v1.md), §1, §4.1, §5; run ID `ham10000-hf-dinov2-primary-v1`.
- EXP-002 — [report](../experiments/EXP-002-ham10000-jepa-hardened-proxy-v1.md), §1, §4.1, §5; run ID `ham10000-hf-dinov2-exp002-v1`.
- EXP-003 — [report](../experiments/EXP-003-ham10000-jepa-held-out-nuisance-v1.md), §1, §4.1, §5; run ID `ham10000-hf-dinov2-exp003-v1`.
- EXP-004 — [report](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md), §1, §4.1, §5; run ID `ham10000-hf-dinov2-exp004-v1`.
- EXP-005 — [report](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md), §1, §4.4 (the underfit diagnosis), §5; run ID `ham10000-hf-dinov2-exp005-v1`.
- EXP-006a — [report](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md), §1, §4, §5; run ID `ham10000-hf-dinov2-exp006a-v1`.
- EXP-006b — [report](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md), §1, §4, §5; run ID `ham10000-hf-clip-exp006b-v1`.
- EXP-007 — [report](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md), §1, §2.3 (contamination), §4, §5, §6; run ID `ham10000-hf-dermlip-exp007-v1`.
- EXP-008 — [report](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md), §1, §2.2 (BiomedCLIP non-contamination), §4.3 (three-way table), §5, §6; run ID `ham10000-hf-biomedclip-exp008-v1`.
- Seed sweep — [summary](../experiments/EXP-007-008-seed-sweep-summary.md), §3, §4, §6; 10 run IDs.

Writing tasks:

1. For each subsection, write a 4–6 sentence paragraph that compresses the EXP §1 Summary into ~80 words. Headline number, scaffold/backbone, one-sentence interpretation.
2. Insert one cross-reference to the report at the end of each paragraph (e.g., "Full breakdown in [EXP-007 §5]").
3. Decide whether §5 should chronologically follow the experiment IDs or be re-ordered into a logical narrative (recommendation: follow IDs, since the falsification ladder is itself the contribution).

### §6. Results (≈1.0 page, dominated by tables and figures)

Content:

- **Table 1.** Cross-run results, expanded with bootstrap CIs and seed std.
- **Figure 1.** Test-AUROC vs pretraining-data domain.
- **Figure 2.** Train-vs-test AUROC scatter across runs 4–8.
- **Figure 3.** Stable / changing pair-score histograms on test split for EXP-006b, EXP-008, EXP-007.
- **Table 2.** Predictor-class × optimiser ablation on DINOv2 ViT-B/14.
- **Table 3.** Seed sweep.
- **Representation-health table.** Appendix.

Source pointers:

- Cross-run table: [EXP-008 §9.4](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) is the most up-to-date version; supplement with seed-mean values from [the seed-sweep summary §4](../experiments/EXP-007-008-seed-sweep-summary.md).
- Figure 1 source values: §1 / §4.1 of every EXP-006b / 007 / 008 report.
- Figure 2 source values: §4.2 / §4.3 of EXP-004 / 005 / 006a / 006b / 007 / 008 reports.
- Figure 3 source images: `outputs/hf-runs/<run-id>/artifacts/plots/jepa_score_histogram.png` for the three relevant run IDs (pulled with `derma-jepa hf-run summary`).
- Table 2 (predictor × optimiser) source: §4.1 of EXP-004 / EXP-005 / EXP-006a.
- Table 3 (seed sweep) source: [seed-sweep summary §3](../experiments/EXP-007-008-seed-sweep-summary.md).
- Representation health: §4.5 of every report; aggregator: write a small script that pulls `representation_health` from each `metrics.json`.

Writing tasks:

1. Render Figure 1 from the cross-run table. Tool: matplotlib, simple barh; annotate the DermLIP bar with "[contamination caveat]".
2. Render Figure 2 (train vs test AUROC scatter). Each point labelled with its EXP ID. Diagonal reference line. Tool: matplotlib.
3. Build Figure 3 by pulling the three histogram PNGs from the run archive and assembling a 3-panel figure. Tool: ImageMagick `montage` or matplotlib subplots that re-load from the underlying score arrays.
4. Build Table 1 from the EXP-008 §9.4 markdown table; add bootstrap-CI columns from the per-run reports; verify against Section 5 of this outline.
5. Render Table 2 from EXP-004 / 005 / 006a §4.1.
6. Pull representation-health values into Appendix Table 4 via a small script.

### §7. Analysis and discussion (≈1.0 page)

Content:

- **§7.1 Why the inversion is below random rather than at chance.** Family-specific direction extrapolation; opposite-sign generalisation rather than uncorrelated.
- **§7.2 Why DermLIP transfers under this scaffold and BiomedCLIP doesn't (with explicit caveat).** Hypothesis: nuisance-direction alignment in the embedding space. Caveat: this hypothesis is not yet distinguishable from "DermLIP saw HAM10000."
- **§7.3 Loss vs AUROC.** MSE delta does not predict train AUROC across optimisers; example: EXP-006a vs EXP-004.
- **§7.4 What is not yet claimed.** Frozen DermLIP "solves" the proxy in a transfer sense — only that DermLIP + linear scaffold reaches AUROC 0.944 ± 0.003 on `strong_held_out_2` under the specific evaluation protocol, with HAM10000 contamination unpartitioned.

Source pointers:

- §7.1 below-random-rather-than-chance argument: [EXP-006b §4.4](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md) (mechanical account), [EXP-006a §4.4](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) (linear-MLP convergence on test), and pair-score histograms in §4.6 of each report.
- §7.2 nuisance-direction-alignment hypothesis: [EXP-007 §4.4](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md), [EXP-008 §4.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).
- §7.3 loss-vs-AUROC decoupling: [EXP-006a §4.3](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) (cross-run loss-trajectory table); per-run training reports under `outputs/hf-runs/<run-id>/artifacts/reports/jepa_training_report.json`.
- §7.4 contamination caveat: [EXP-007 §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md), [EXP-008 §5.1](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).

Writing tasks:

1. Draft §7.1 (~150 words) referencing the histogram pattern from EXP-004 §4.6 / EXP-006b §4.6.
2. Draft §7.2 (~200 words). Mention that a direction-structure probe (PCA of stable-pair difference vectors per family) would test the hypothesis directly; flag as future work or run before submission if time permits.
3. Draft §7.3 (~120 words) using the EXP-006a vs EXP-004 loss-delta contrast.
4. Draft §7.4 (~100 words). This paragraph is the hardest to write honestly; do a separate review pass with a critic-mode read on it.

### §8. Required follow-up work — EXP-009 (≈0.5 page)

Content:

- **§8.1 The partition question.** What's unresolved: dermoscopy-domain transfer vs HAM10000 image-level overlap.
- **§8.2 EXP-009 design.** Self-pretrain a DINOv2 ViT-B/14 on a non-HAM10000 dermoscopy corpus (ISIC archives minus HAM10000, DermNet, BCN20000 non-HAM10000 components). Short JEPA-style or MIM objective (~20 epochs). Run the EXP-004 recipe on top.
- **§8.3 Decision table for EXP-009 outcome.** Three outcome bands → three paper-claim revisions.
- **§8.4 Why EXP-009 was not run before submission.** Corpus-assembly cost (provenance audit on every ISIC component, SSL pretrain compute, pretrain-quality check on a held-out HAM10000 split). Choice to share methodology and the nine-run arc at this stage rather than block.
- **§8.5 Feedback being sought.** Whether the proxy-task design is reasonable; whether the three-family disjointness is the right granularity; whether the proposed partition is the right one or whether a different partition (e.g. a dermoscopy-text-only retrieval probe over DermLIP) would be more informative; what other follow-ups would matter for the next stage.

Source pointers:

- EXP-009 design (current draft): [EXP-008 §7](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) and [EXP-007 §7](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).
- Decision table (current draft): [EXP-008 §7](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).
- Open partition discussion: [EXP-007 §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md), [EXP-008 §5.1, §5.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).

Writing tasks:

1. Pull the decision table from EXP-008 §7 and adapt to a paper-quality format (3 rows, AUROC band column, interpretation column, claim-revision column).
2. Draft §8.4 honestly: corpus assembly is non-trivial work, the methodology is worth getting feedback on first, the paper is honest about not running it.
3. Draft §8.5 as a numbered list of 4–6 specific feedback questions the reader can respond to.

### §9. Limitations (≈0.5 page)

1. **Pretraining contamination is unpartitioned.** Central caveat. EXP-009 (§8) addresses it.
2. **Synthetic nuisance.** `strong`, `strong_held_out`, `strong_held_out_2` are deterministic operations, not real-world acquisition variability.
3. **Single-architecture comparison.** All four backbones are ViT-B-class. Other architectures might behave differently.
4. **Single dataset.** HAM10000 only.
5. **HAM10000 is cross-sectional.** No real same-lesion-over-time pairs; the proxy is synthetic throughout.
6. **One scaffold class on the dermoscopy backbone.** Only linear; MLP-on-DermLIP not run.

Source pointers:

- §1 (contamination): [EXP-007 §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md) #1.
- §2–6: [EXP-008 §6](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md), [EXP-007 §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).

Writing tasks:

1. Draft each as one short sentence + one short justification sentence. Total ~300 words.

### §10. Conclusion (≈0.25 page)

Content: restate the headline (pretraining-data domain is the load-bearing axis under this protocol, with caveats); restate that the paper has not validated or invalidated the underlying thesis; point at EXP-009 and at real longitudinal data as the natural extensions.

Source pointers:

- Headline: [EXP-008 §5.2](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) (updated thesis statement).
- Honest non-claim: [EXP-007 §5.1](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md) (the four "proved / not proved" bullets).

Writing tasks:

1. Draft a ~200-word conclusion. End with the feedback ask.

### §11. Contributions (target placement: bullet list at end of §1)

1. **Methodological.** Leakage-controlled HAM10000 longitudinal-proxy task with three disjoint synthetic nuisance families. Released as code + configs + reproduction launchers.
2. **Empirical (negative).** Across two architectures × three predictor scaffolds × two optimisers, frozen natural-image backbones produce below-random inverted test AUROC on the held-out third family.
3. **Empirical (positive but unpartitioned).** Frozen DermLIP reaches test AUROC 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline. The win cannot yet be attributed to dermoscopy-domain transfer rather than HAM10000 image-level overlap; EXP-009 (§8) is the open partition.
4. **Empirical (negative-as-partition).** Frozen BiomedCLIP, the cleanest publicly available "general medical pretraining without HAM10000" backbone, lifts only +0.04 over web CLIP, ruling out "any medical-image pretraining" as sufficient.
5. **Reproducibility.** Every primary-tier run is archived publicly with manifests, embeddings, metrics, model card, and logs at `abdelstark/derma-jepa-runs/<run_id>`. A single launcher script reproduces each run end-to-end on a free A10G via Hugging Face Jobs.
6. **A clearly-scoped open question.** EXP-009's design, decision table, and place in the paper are stated up front rather than treated as future work in passing.

Writing tasks:

1. Convert this list into a 5-line bulleted block at the end of §1. Drop §11 from the section structure once the contributions are in §1.

---

## 4. Master figure / table list (with build instructions)

| Asset | Source | Renderer | Notes |
|---|---|---|---|
| Table 1: Cross-run results | [EXP-008 §9.4](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) + [seed-sweep summary §4](../experiments/EXP-007-008-seed-sweep-summary.md) | LaTeX `tabular`/`booktabs` | DermLIP row carries explicit "contamination caveat" annotation. |
| Figure 1: Test AUROC vs pretraining domain | New panel from cross-run table | matplotlib barh | DermLIP bar annotated. |
| Figure 2: Train-vs-test AUROC scatter | Per-run `metrics.json` (pull from run archive) | matplotlib | Diagonal reference line; runs 4–8 labelled. |
| Figure 3: 3-panel JEPA score histograms | `outputs/hf-runs/<run-id>/artifacts/plots/jepa_score_histogram.png` for EXP-006b / EXP-007 / EXP-008 | matplotlib subplots from underlying score arrays in `predictor_latents.npz` | Tighter typography in paper than the run-archive PNGs. |
| Table 2: Predictor × optimiser | EXP-004 §4.1, EXP-005 §4.1, EXP-006a §4.1 | LaTeX | |
| Table 3: Seed sweep | [Seed-sweep summary §3](../experiments/EXP-007-008-seed-sweep-summary.md) | LaTeX | |
| Table 4: Representation health | §4.5 of every report | LaTeX | Appendix only. |
| Pipeline diagram | [README Mermaid § Pipeline overview](../../README.md) | TikZ or Inkscape-exported PDF | |
| Three-family experimental design | [README Mermaid § Experimental design](../../README.md) | TikZ or Inkscape | |
| Predictor-objective diagram | [README Mermaid § Method](../../README.md) | TikZ or Inkscape | |
| EXP-009 decision-table | §8.3 above | LaTeX | |

For paper rendering, the README Mermaid sources should be re-rendered as TikZ or vector PDF before submission. Mermaid is fine for the GitHub README but not for camera-ready.

---

## 5. Numbers safe to cite (single source of truth)

Pulled from the locked seed-sweep summary and per-run reports. Every number cited in the paper must come from this list. Updates to this list propagate to the abstract, §1, §5, §6, and the conclusion.

- DermLIP-linear test AUROC: **0.944 ± 0.003** (n = 5 seeds, range 0.939–0.947, 95 % CI[mean] [0.941, 0.946]) — [seed-sweep §3](../experiments/EXP-007-008-seed-sweep-summary.md)
- BiomedCLIP-linear test AUROC: **0.329 ± 0.012** (n = 5 seeds, range 0.312–0.344, 95 % CI[mean] [0.318, 0.339]) — [seed-sweep §3](../experiments/EXP-007-008-seed-sweep-summary.md)
- OpenAI CLIP-linear test AUROC: 0.286 [0.265, 0.310] (n = 1) — [EXP-006b §4.1](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md)
- DINOv2 B/14-linear test AUROC: 0.249 [0.230, 0.270] (n = 1, EXP-004) — [EXP-004 §4.1](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md)
- DINOv2 B/14-MLP-Adam test AUROC: 0.248 [0.228, 0.269] (n = 1, EXP-006a) — [EXP-006a §4.1](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md)
- Pixel L2 baseline test AUROC: 0.580 [0.556, 0.606] (deterministic, identical across runs) — any post-EXP-004 report §4.1
- SSIM distance test AUROC: 0.436 [0.411, 0.459]
- Web → general-medical step: **+0.04** AUROC
- General-medical → dermoscopy step: **+0.62** AUROC
- Train → test drop, EXP-007: −0.05
- Train → test drop, EXP-006b: −0.70
- DermLIP raw cosine baseline: 0.109 [0.095, 0.124]
- BiomedCLIP raw cosine baseline: 0.047 [0.040, 0.055]
- OpenAI CLIP raw cosine baseline: 0.036 [0.030, 0.043]

---

## 6. Reference list (target ≈25 entries)

Already in [`README.md` § References](../../README.md). Add for the paper:

- Caron, M. et al. *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021. arXiv:2104.14294.
- He, K. et al. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. arXiv:2111.06377.
- Bardes, A., Ponce, J., LeCun, Y. *VICReg*. ICLR 2022. arXiv:2105.04906.
- Gulrajani, I. and Lopez-Paz, D. *In Search of Lost Domain Generalization*. ICLR 2021. arXiv:2007.01434.
- Koh, P. W. et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML 2021. arXiv:2012.07421.
- Codella, N. et al. ISIC 2017 / 2018 / 2019 challenge papers.
- Combalia, M. et al. *BCN20000: Dermoscopic Lesions in the Wild*. arXiv:1908.02288 (cited for EXP-009 corpus design).
- Bommasani, R. et al. *On the Opportunities and Risks of Foundation Models*. arXiv:2108.07258 (cited in framing of contamination risks).
- The eight primary citations already in the README: I-JEPA, OpenCLIP, JEPA position paper, DINOv2, CLIP, HAM10000, DermLIP/Derm1M, BiomedCLIP.

---

## 7. Appendix structure

- **A. Per-run details.** One subsection per primary-tier run, summarising the experiment report's §1 / §4 / §5 / §6.
- **B. Hyperparameters.** Predictor architecture, optimiser, LR, weight decay, batch size, epochs, identity warm-start; per variant.
- **C. Nuisance-family operation specifications.** Deterministic operation lists for `strong`, `strong_held_out`, `strong_held_out_2`. Source: [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py).
- **D. Bootstrap CI protocol.** Per-group resampling, 1,000 samples, 95 % bands. Source: [`src/derma_jepa/metrics.py`](../../src/derma_jepa/metrics.py); validation: [`tests/test_metrics.py`](../../tests/test_metrics.py).
- **E. Representation-health checks.** Prediction-norm, dimension-variance, collapse flag definitions and per-run values.
- **F. Compute budget.** Per-run wall-clock breakdown; total GPU-hours across all 9 runs + seed sweep.
- **G. Reproducibility checklist.** Following NeurIPS / MLRC standard items, mapped to repository assets.
- **H. Pretraining contamination analysis.** Detailed argument for why HAM10000 is in Derm1M and the boundaries of what EXP-007 / EXP-008 can and cannot conclude. Load-bearing for the paper.
- **I. EXP-009 design (planned).** Corpus assembly plan, SSL pretrain protocol, decision table mapping outcome to claim revision.

---

## 8. Submission targets

In rough order of fit, prioritising venues that accept honest preliminary reports:

| Venue | Track / format | Fit | Notes |
|---|---|---|---|
| arXiv preprint | non-archival | always first | Recommended ahead of any venue submission to lock priority date and to gather feedback. |
| TMLR | full paper, rolling | high | Empirical / methodology track, no page-limit pressure, accepts honest preliminary work. Probably the best primary venue for this paper. |
| MIDL 2026 | short paper (4 pp + refs) | high | Short-paper format is well-suited to the preliminary-evidence framing. |
| ML4H 2026 (NeurIPS workshop) | full or extended-abstract | high | Health-ML community feedback is exactly what's wanted at this stage. |
| MICCAI 2026 (workshop, e.g. ISIC, MILLanD) | workshop paper | medium-high | Skin-imaging audience. Full MICCAI track is harder without EXP-009. |
| MIDL 2026 (full) | full paper | medium | Full-paper format probably wants EXP-009 done; defer to v2 if EXP-009 lands. |
| NeurIPS 2026 Datasets & Benchmarks | full paper | medium | Methodological contribution and run archive fit, but reviewers will press on contamination. Better as a v2 venue after EXP-009. |
| Full NeurIPS / ICLR | full paper | low for this paper | Wait until EXP-009 closes the partition. |

Concrete plan: arXiv preprint now, TMLR or ML4H workshop submission within 4 weeks, then a v2 with EXP-009 results for a MIDL or MICCAI full-paper submission.

---

## 9. Open issues for the writing pass

- Whether the abstract should lead with the negative-result-arc framing or with the DermLIP-specific positive headline. Current outline leads with the gradient and front-loads the contamination caveat. Both framings are honest; reviewer preferences differ.
- Whether to include the BiomedCLIP partition in §5 alongside DermLIP or to give it a dedicated subsection. Outline currently uses dedicated subsections (§5.7 and §5.8).
- Whether to run a direction-structure probe (§7.2) before submission. It would let the family-alignment hypothesis become a result rather than a hypothesis. Compute-cheap; recommend yes if time permits.
- How much of the appendix to ship in v1. For a preliminary preprint, all nine appendices in §7 are useful; for a workshop short paper, B / D / G / H are essential and the rest can move to a supplement repo.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-28 | Abdelhamid Bakhta | Initial outline. |
| 2026-04-28 | Abdelhamid Bakhta | Reframed as preliminary release without EXP-009. |
| 2026-04-28 | Abdelhamid Bakhta | Deepening pass: per-section source pointers (code, configs, scripts, tests, experiment-report sections), explicit writing tasks per section, master figure/table build instructions, and a separate companion writing plan at [`docs/paper/WRITING-PLAN.md`](WRITING-PLAN.md). |
