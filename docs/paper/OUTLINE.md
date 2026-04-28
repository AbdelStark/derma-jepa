# DermaJEPA paper outline (v3, preliminary release, code-grounded)

**Author:** Abdelhamid Bakhta (sole)
**Status:** Working outline for a preliminary preprint. Each subsection now carries source pointers verified against the codebase (function name + line range), equation drafts in LaTeX, sample paragraph drafts for the hardest sections, and a per-section word budget. EXP-009 is **not** in scope; it is reported as required follow-up with a decision table.
**Source repository:** [github.com/AbdelStark/derma-jepa](https://github.com/AbdelStark/derma-jepa)
**Run archive:** [`abdelstark/derma-jepa-runs`](https://huggingface.co/datasets/abdelstark/derma-jepa-runs)
**Companion documents:**
- [`docs/paper/WRITING-PLAN.md`](WRITING-PLAN.md) — operational checklist (phases, build commands, submission cascade).
- [`docs/paper/REVIEWER-QA.md`](REVIEWER-QA.md) — anticipated reviewer questions with prepared responses.

**LaTeX scaffold:** [`paper/`](../../paper/) — `main.tex`, `refs.bib` (16 entries), section files (`sections/00-abstract.tex` through `sections/10-conclusion.tex` plus 9 appendix stubs), `figures/locked-numbers.json` (single source of truth for every quantitative claim), `figures/build_fig1.py`, `tables/tab1-cross-run.tex`, and `verify_claims.py` (cross-checks every locked number against the run-archive mirror; passes 18/18 at delta 0.0000). Build via `cd paper && make verify && make figures && make pdf`.

---

## 0. Framing

This paper is a *preliminary report*. Sharing the arc at this stage is justified by:

1. The methodology — a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families and a `train-on-two / evaluate-on-third` protocol — is novel and deserves community feedback before further compute is spent extending it.
2. The nine-run sequence already characterises a clean failure mode of frozen natural-image and frozen general-medical backbones (test AUROC inverts below random across two architectures, three predictor scaffolds, two optimisers).
3. The single positive result observed (frozen DermLIP at 0.944 ± 0.003 across 5 seeds) **cannot be attributed to dermoscopy-domain transfer with the experiments in this paper alone**. DermLIP's pretraining corpus, Derm1M, almost certainly contains HAM10000 raw images, and the BiomedCLIP partition (EXP-008) only rules out the broader "any medical-image pretraining" alternative — not HAM10000 image-level overlap specifically.
4. EXP-009 (a self-pretrained DINOv2 on a non-HAM10000 dermoscopy corpus) is the experiment that would partition dermoscopy-domain transfer from contamination. Building that corpus and running the SSL pretrain is multi-week engineering. Releasing the methodology now to validate the proxy design and gather feedback on the partition before committing to it.

**Source pointers for the framing claims (file:line · symbol):**

- Three-family disjointness: [`src/derma_jepa/public_data.py:494` `_apply_strong_held_out_2_nuisance`](../../src/derma_jepa/public_data.py), [`src/derma_jepa/public_data.py:751` `_apply_strong_nuisance`](../../src/derma_jepa/public_data.py), [`src/derma_jepa/public_data.py:829` `_apply_strong_held_out_nuisance`](../../src/derma_jepa/public_data.py).
- Train-on-two / evaluate-on-third dispatch: [`src/derma_jepa/public_data.py:568` `_severity_for_split`](../../src/derma_jepa/public_data.py), [`src/derma_jepa/public_data.py:584` `_parse_severity_list`](../../src/derma_jepa/public_data.py), and the canonical proxy config [`configs/data/ham10000_hf_mounted_exp004.yaml`](../../configs/data/ham10000_hf_mounted_exp004.yaml).
- Below-random inversion claim: [EXP-004 §4](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md) (DINOv2 linear), [EXP-005 §4](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md) (MLP under SGD), [EXP-006a §4](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) (MLP under Adam), [EXP-006b §4](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md) (OpenAI CLIP linear).
- DermLIP positive + caveat: [EXP-007 §1, §6](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).
- BiomedCLIP partition: [EXP-008 §1, §4.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).
- Seed sweep: [seed-sweep summary §3, §4](../experiments/EXP-007-008-seed-sweep-summary.md).

---

## 1. Title

Three working titles, all explicit about the preliminary status:

1. **Frozen-backbone JEPA-style probes on HAM10000: preliminary results from a nine-experiment ablation, with an unpartitioned contamination caveat.**
2. *Toward a clean test of frozen-backbone JEPA generalisation on dermoscopy: methodology, early evidence, and an open partition experiment.*
3. *When does frozen-backbone JEPA generalise on dermoscopy? A preliminary nine-experiment study and a planned partition.*

Final title locks at submission. The "preliminary" qualifier stays in the title regardless. Recommendation: option 1 unless reviewer feedback objects.

---

## 2. Abstract (≈260 words)

Self-supervised representations promise robustness to nuisance variation; whether that robustness transfers across novel nuisance families is rarely tested. We probe this on a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families — partitioned along the deterministic-augmentation axis — training a JEPA-style latent predictor on stable pairs from two of three families and evaluating on the third unseen family. We report a preliminary nine-experiment arc. Across two frozen natural-image backbones (DINOv2 ViT-B/14, OpenAI CLIP ViT-B/16), three predictor scaffolds (linear, underfit MLP, fit MLP under Adam), and two optimisers (SGD, Adam), test AUROC on the held-out family is 0.25–0.29 — below random and below cheap baselines (pixel L2, SSIM, raw embedding cosine). A frozen general-medical backbone (BiomedCLIP, PMC-15M, no HAM10000) lifts test AUROC by only +0.04 to 0.329 ± 0.012 across 5 seeds. A frozen dermoscopy-pretrained backbone (DermLIP, Derm1M) lifts test AUROC to 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline and the only above-random result on the held-out family across nine runs. Holding architecture (ViT-B/16) and predictor (linear) constant, varying only the pretraining-data domain produces a +0.66 AUROC swing localised to the dermoscopy-specific step. **The paper explicitly does not claim that this swing is dermoscopy-domain transfer**: DermLIP's Derm1M corpus almost certainly contains HAM10000, and the experiments reported here cannot separate dermoscopy-domain transfer from HAM10000 image-level overlap. The methodology, configurations, and run archive are released, and a planned partition experiment (a non-HAM10000 dermoscopy SSL pretrain) is presented as required follow-up. The contribution is the proxy-task design, the failure characterisation, and a clearly scoped open question — not a conclusion about whether frozen-backbone JEPA generalises on dermoscopy.

---

## 3. Section structure

Each subsection lists (i) what the section says, (ii) source pointers verified against the code at file:line · symbol, (iii) writing tasks, (iv) word budget.

### §1. Introduction (≈1.0 page; ≈600 words; ≈10 paragraphs of ~60 words)

Content blocks (target word counts):

| Block | Words | What it says |
|---|---:|---|
| Motivation | 120 | Lesion monitoring is a change-detection problem under heavy nuisance. SSL representations are pitched as nuisance-invariant; the question is whether that invariance generalises to *unseen* nuisance families. |
| Setting | 80 | HAM10000 is cross-sectional. We construct a leakage-controlled longitudinal proxy: lesion-ID splits, synthetic stable-pair augmentations applied post-split, changing-pair construction matched on diagnosis and anatomical site. |
| Probe | 100 | Linear or 2-layer identity-residual MLP over a frozen vision backbone, trained to minimise L2 between predicted and observed target latents on stable pairs. Cite the JEPA family. |
| Ablation framing | 60 | Nine experiments. Predictor class, optimiser, and backbone are varied independently while every other knob is held constant. |
| Preliminary status | 90 | This is an early-stage report. The DermLIP positive is unpartitioned vs HAM10000 contamination. EXP-009 is required to close it. We share now to validate the methodology and to invite feedback on the partition design. |
| Contributions list | 90 | Six bullets (mirrored in §11 below; here as a list at the end of §1). |
| Headline figure call-out | 60 | "Figure 1 shows the cross-backbone gradient: web (0.286), general-medical (0.329 ± 0.012), dermoscopy (0.944 ± 0.003); the dermoscopy bar carries the contamination caveat." |

Source pointers:

- Lesion monitoring framing: [`docs/prd/PRD.md` §3](../prd/PRD.md), [`docs/spec/MVP-SPEC.md` §1](../spec/MVP-SPEC.md), [`docs/rfcs/RFC-0001-problem-statement-and-product-thesis.md`](../rfcs/RFC-0001-problem-statement-and-product-thesis.md).
- "HAM10000 is cross-sectional": [Tschandl et al., *Sci. Data* 2018](https://doi.org/10.1038/sdata.2018.161); see also [`data/README.md`](../../data/README.md).
- Lesion-ID splits: [`src/derma_jepa/public_data.py:326` `_split_records`](../../src/derma_jepa/public_data.py), [`src/derma_jepa/public_data.py:1222` `_split_group_id`](../../src/derma_jepa/public_data.py); test: [`tests/test_public_data_pipeline.py`](../../tests/test_public_data_pipeline.py).
- Stable-pair augmentation: [`src/derma_jepa/public_data.py:674` `_write_stable_variant`](../../src/derma_jepa/public_data.py).
- Changing-pair construction: [`src/derma_jepa/public_data.py:609` `_match_changing_target`](../../src/derma_jepa/public_data.py); strict same-diagnosis-site policy at lines 623–630.
- Predictor objective: [`src/derma_jepa/training.py:26` `train_jepa_predictor`](../../src/derma_jepa/training.py); the L2-MSE loss at [`src/derma_jepa/training.py:402` `_loss`](../../src/derma_jepa/training.py).

Writing tasks:

1. Draft motivation paragraph (~120 words) without clinical-utility promises.
2. Draft setting paragraph (~80 words) naming HAM10000 as cross-sectional and the proxy as synthetic.
3. Draft probe paragraph (~100 words) defining JEPA-style narrowly.
4. Draft preliminary-status paragraph (~90 words) — caveat named, EXP-009 named, feedback ask explicit.
5. Insert Figure 1 reference + one-sentence reading instruction.

### §2. Related work (≈0.75 page; ≈450 words; ≈3 paragraphs of ~150 words)

Content (one paragraph each):

- **JEPA-style and self-supervised image representations.** Position the probe as "the frozen-backbone, single-step, stable-pair subset of the JEPA family applied as an evaluation probe rather than a pretraining objective." Cite [LeCun, 2022], [Assran et al., 2023] (I-JEPA), [Caron et al., 2021] (DINO), [Oquab et al., 2024] (DINOv2), [Radford et al., 2021] (CLIP), [Cherti et al., 2023] (OpenCLIP).
- **Domain-specific medical foundation models.** [Zhang et al., 2023] (BiomedCLIP), [Yan et al., 2025] (DermLIP / PanDerm), MedCLIP, MONET. State which weights are used and which are referenced for context.
- **Distribution-shift evaluation in dermoscopy / medical imaging.** ISIC challenge series, RETFound and similar foundation-model evaluations, [Koh et al., 2021] (WILDS), [Gulrajani-Lopez-Paz, 2021] (in-search-of-lost-DG). Position the contribution as the deterministic, leakage-controlled, three-family-disjoint nuisance design.

Source pointers:

- DINOv2 loader: [`src/derma_jepa/embeddings.py:181` `_dinov2_matrix`](../../src/derma_jepa/embeddings.py).
- OpenAI CLIP loader: [`src/derma_jepa/embeddings.py:233` `_clip_matrix`](../../src/derma_jepa/embeddings.py).
- DermLIP / BiomedCLIP loader: [`src/derma_jepa/embeddings.py:285` `_open_clip_matrix`](../../src/derma_jepa/embeddings.py).
- BibTeX skeleton: [`docs/paper/WRITING-PLAN.md` §5](WRITING-PLAN.md).

Writing tasks:

1. Re-read I-JEPA [arXiv:2301.08243] and the JEPA position paper to refresh framing.
2. Read BiomedCLIP §3 [arXiv:2303.00915] for the PMC-15M description supporting the "no HAM10000 in pretraining" claim.
3. Read DermLIP / Derm1M §3 [arXiv:2503.14911] to confirm absence of a per-source breakdown; cite that absence as the basis for "almost certainly includes HAM10000."
4. Skim WILDS [arXiv:2012.07421] and Gulrajani-Lopez-Paz [arXiv:2007.01434] for the held-out-family evaluation framing.

### §3. Method (≈1.0 page; ≈600 words; equations + 4 short subsections)

#### §3.1 Notation and probe

Let $f : \mathcal{X} \to \mathbb{R}^d$ be a frozen vision encoder with output dimension $d$, and define the L2-normalised representation

$$z(x) \;=\; \frac{f(x)}{\lVert f(x) \rVert_2} \;\in\; \mathbb{S}^{d-1}.$$

A *stable* pair $(x_c, x_t)$ shares a single underlying lesion image $u$ subjected to two independent draws of a synthetic nuisance augmentation $\nu \sim \mathcal{N}_F$ from family $F$:

$$x_c = \nu_c(u), \qquad x_t = \nu_t(u), \qquad \nu_c, \nu_t \sim \mathcal{N}_F.$$

A *changing* pair pairs two distinct lesions $u_1, u_2$ matched on diagnosis $\mathrm{dx}(u_1) = \mathrm{dx}(u_2)$ and anatomical site $\mathrm{site}(u_1) = \mathrm{site}(u_2)$; each side receives an independent nuisance draw.

The predictor $g_\theta : \mathbb{R}^d \to \mathbb{R}^d$ minimises the L2-MSE objective on stable pairs in the training split, with a soft `weight − I` regulariser to keep the function close to identity at initialisation:

$$\mathcal{L}(\theta) \;=\; \frac{1}{|\mathcal{D}_{\mathrm{train}}^{\mathrm{stable}}|} \sum_{(x_c, x_t) \in \mathcal{D}_{\mathrm{train}}^{\mathrm{stable}}} \lVert g_\theta(z(x_c)) - z(x_t) \rVert_2^{2} \;+\; \lambda \, \lVert W - I \rVert_F^{2},$$

where $\lambda$ is the configured weight decay and $W$ is the linear weight (or, for the MLP, the residual weight at appropriate index).

At evaluation time, the score for a pair is the cosine *distance* between the L2-normalised prediction and the L2-normalised target:

$$s(x_c, x_t) \;=\; 1 \;-\; \langle \tilde{g}_\theta(z(x_c)), \, z(x_t) \rangle, \qquad \tilde{g}_\theta(\cdot) := \frac{g_\theta(\cdot)}{\lVert g_\theta(\cdot) \rVert_2}.$$

A correctly oriented predictor produces $s(x_c, x_t) < s(x_c', x_t')$ whenever $(x_c, x_t)$ is stable and $(x_c', x_t')$ is changing; AUROC quantifies this ranking.

#### §3.2 Predictor variants

- **Linear with identity warm-start.** $g_\theta(z) = (W) z + b$ with $W$ initialised to $I + \epsilon$, $\epsilon_{ij} \overset{\text{iid}}{\sim} \mathcal{N}(0, 0.005^2)$, and $b = 0$. Under SGD with the soft `weight − I` penalty above, the predictor at step zero is identity-equivalent. Source: [`src/derma_jepa/training.py:216` `_fit_linear_predictor`](../../src/derma_jepa/training.py) (lines 224–225 init; line 242 regulariser; line 244 SGD step).
- **2-layer identity-residual MLP.** $g_\theta(z) = z + W_2 \,\mathrm{ReLU}(W_1 z + b_1) + b_2$, with $W_2$ zero-initialised so the residual term vanishes at step zero and $g_\theta = \mathrm{identity}$. Hidden dim 512. Source: [`src/derma_jepa/training.py:277` `_fit_mlp_predictor`](../../src/derma_jepa/training.py); docstring at lines 284–289 documents the identity warm-start.

#### §3.3 Three nuisance families

Operation lists, verified against [`src/derma_jepa/public_data.py`](../../src/derma_jepa/public_data.py):

| Family | Source line | Operations |
|---|---|---|
| `strong` | line 751 (`_apply_strong_nuisance`) | brightness × contrast × saturation, rotation $\pm 15°$, scale $0.82$–$1.00$, translate $\pm 5\%$, horizontal flip, Gaussian blur, Gaussian noise, JPEG quality $45$–$70$ |
| `strong_held_out` | line 829 (`_apply_strong_held_out_nuisance`) | hue shift $\pm 12°$ (HSV rotation), posterise $4$–$6$ bits, unsharp-mask sharpen, motion blur (linear kernel $5$–$15\,\mathrm{px}$, vertical or horizontal), random rectangular erasing $6$–$18\%$ area, JPEG quality $20$–$40$ |
| `strong_held_out_2` | line 494 (`_apply_strong_held_out_2_nuisance`) | gamma $0.6$–$1.6$, colour-temperature shift $0.85$–$1.15$ (channel rescale), radial vignette $25$–$55\%$ falloff, salt-and-pepper noise $0.5$–$2.5\%$, JPEG quality $80$–$95$ |

The disjointness rule is operation-level (no shared transform type between any two families), enforced by inspection of the three function bodies. The augmentation suite source code is committed at the line numbers above.

#### §3.4 Splits and pair construction

- Lesion-ID-aware splits at fractions $0.70 / 0.15 / 0.15$, deterministic from `seed: 20260422`, group sizes $5{,}229 / 1{,}120 / 1{,}121$ across HAM10000's $7{,}470$ unique lesions. Source: [`src/derma_jepa/public_data.py:326` `_split_records`](../../src/derma_jepa/public_data.py).
- $1{,}000$ stable + $1{,}000$ changing pairs per split (total $6{,}000$ pairs per run).
- Strict same-diagnosis-site changing-pair policy: a context lesion's changing partner must be a distinct lesion with identical diagnosis and identical anatomical site. Source: [`src/derma_jepa/public_data.py:609` `_match_changing_target`](../../src/derma_jepa/public_data.py) lines 623–630.
- Train stable pairs rotate between `strong` and `strong_held_out` deterministically by pair index. Source: [`src/derma_jepa/public_data.py:674` `_write_stable_variant`](../../src/derma_jepa/public_data.py) line 685.
- Val and test stable pairs use `strong_held_out_2` only. Source: [`src/derma_jepa/public_data.py:568` `_severity_for_split`](../../src/derma_jepa/public_data.py).

#### §3.5 Evaluation protocol

- AUROC primary, computed via [`src/derma_jepa/metrics.py:42` `roc_auc_score`](../../src/derma_jepa/metrics.py) wrapping scikit-learn.
- Bootstrap 95 % CIs over groups: $1{,}000$ resamples, group-level (pair-level) resampling. Source: [`src/derma_jepa/metrics.py:66` `bootstrap_auroc_ci`](../../src/derma_jepa/metrics.py).
- AUPRC, equal-error-rate threshold, and FPR at fixed TPR $= 0.8$ accompany every run. Source: [`src/derma_jepa/metrics.py:29` `binary_metric_summary`](../../src/derma_jepa/metrics.py).
- Three representation-health checks per run: prediction-norm mean / min, dimension-variance mean / min, and a binary `collapsed` flag triggered if either falls below $10^{-8}$. Source: [`src/derma_jepa/training.py:479` `_collapse_checks`](../../src/derma_jepa/training.py).
- Tests: [`tests/test_metrics.py`](../../tests/test_metrics.py) covers AUROC, bootstrap CIs, EER, and FPR-at-TPR.

Writing tasks:

1. Drop the equations above into `paper/sections/03-method.tex` verbatim.
2. Render the predictor diagram (Mermaid → TikZ); see [`README.md` § Method](../../README.md).
3. Render the experimental-design diagram (Mermaid → TikZ); see [`README.md` § Experimental design](../../README.md).
4. Reference [`tests/test_metrics.py`](../../tests/test_metrics.py) in §3.5 for reviewer-reproducibility.

### §4. Experimental setup (≈0.5 page; ≈300 words)

| Block | Words | Content |
|---|---:|---|
| Backbones table | 0 (table) | Replicate the README backbones table; add embedding-dim column. |
| Training | 100 | SGD vs Adam dispatch, 200 epochs at batch 128, identity warm-start, hyperparameters per variant. |
| Compute | 80 | Single A10G 24 GB on Hugging Face Jobs; ~85 min wall-clock per run end-to-end. Pinned dependencies via [`scripts/hf_jobs_constraints.txt`](../../scripts/hf_jobs_constraints.txt). |
| Code & archive | 60 | Code at the GitHub repo, runs at the HF dataset, configs under `configs/data/`. |

Source pointers:

- Backbone table: [`README.md` § Backbones evaluated](../../README.md).
- Training config schema: [`src/derma_jepa/config.py`](../../src/derma_jepa/config.py) — `TrainingConfig`, `EmbeddingModelConfig`.
- Compute pins: [`scripts/hf_jobs_constraints.txt`](../../scripts/hf_jobs_constraints.txt).
- Per-run wall-clock: §3 of every experiment report under [`docs/experiments/`](../experiments/README.md).

### §5. Experimental sequence (≈1.0 page; ≈600 words; 9 paragraphs of ~60–70 words)

| Stage | Source report | Headline number to lead with |
|---|---|---|
| §5.1 Sanity | [EXP-001 §1, §4](../experiments/EXP-001-ham10000-jepa-primary-v1.md) | Trivial proxy ceiling at AUROC ≈ 1.000 across all baselines |
| §5.2 Hardened proxy, matched eval | [EXP-002 §1, §4.1](../experiments/EXP-002-ham10000-jepa-hardened-proxy-v1.md) | JEPA at 0.920 vs DINOv2-S = 0.652 (Δ +0.269) |
| §5.3 One-family-held-out | [EXP-003 §1, §4.1](../experiments/EXP-003-ham10000-jepa-held-out-nuisance-v1.md) | JEPA at 0.680 vs SSIM = 0.961 (Δ −0.281) |
| §5.4 Three-family training, third-family eval | [EXP-004 §1, §4.1](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md) | JEPA at 0.249 vs pixel L2 = 0.580 (Δ −0.331) |
| §5.5 Scaffold ablation | [EXP-005 §1, §4](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md), [EXP-006a §1, §4](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) | MLP-SGD train 0.572 / test 0.270; MLP-Adam train 0.893 / test 0.248 |
| §5.6 Backbone ablation, natural-image | [EXP-006b §1, §4](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md) | OpenAI CLIP linear at 0.286 (Δ −0.294) |
| §5.7 Pretraining-domain ablation | [EXP-007 §1, §4](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md) | DermLIP linear at 0.945 (Δ +0.364); contamination caveat introduced |
| §5.8 Pretraining-domain partition | [EXP-008 §1, §4.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) | BiomedCLIP linear at 0.325; web→general-medical +0.04, general-medical→dermoscopy +0.62 |
| §5.9 Seed robustness | [seed-sweep §3, §4](../experiments/EXP-007-008-seed-sweep-summary.md) | DermLIP 0.944 ± 0.003, BiomedCLIP 0.329 ± 0.012 (5 seeds each) |

Writing tasks per stage:

1. Compress the EXP §1 Summary into a 60–70-word paragraph: headline number, scaffold/backbone, one-sentence interpretation, one cross-reference to the full report.
2. Maintain consistent verb tense (past for the experiment, present for the conclusion).
3. End each paragraph with `[full breakdown: EXP-NNN §x]` for navigation.

### §6. Results (≈1.0 page, dominated by tables and figures)

Figure / table inventory (all build instructions in [`docs/paper/WRITING-PLAN.md`](WRITING-PLAN.md) Phase 1):

| Asset | Source data | Build script |
|---|---|---|
| Table 1: Cross-run results | [EXP-008 §9.4](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) + [seed-sweep §4](../experiments/EXP-007-008-seed-sweep-summary.md) | hand-written `paper/tables/tab1-cross-run.tex` |
| Figure 1: Cross-backbone gradient bar plot | locked numbers JSON | `paper/figures/build_fig1.py` |
| Figure 2: Train-vs-test AUROC scatter | per-run `metrics.json` from `outputs/hf-runs/<run-id>/` | `paper/figures/build_fig2.py` |
| Figure 3: 3-panel pair-score histograms | `outputs/hf-runs/<run-id>/artifacts/embeddings/jepa_predictor_latents.npz` for runs `…-clip-exp006b-v1`, `…-dermlip-exp007-v1`, `…-biomedclip-exp008-v1` | `paper/figures/build_fig3.py` |
| Table 2: Predictor × optimiser ablation | [EXP-004 §4.1](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md), [EXP-005 §4.1](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md), [EXP-006a §4.1](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) | hand-written `tab2-predictor-optimiser.tex` |
| Table 3: Seed sweep | [seed-sweep §3](../experiments/EXP-007-008-seed-sweep-summary.md) | hand-written `tab3-seed-sweep.tex` |
| Table 4 (appendix): Representation health | §4.5 of every report | scripted aggregator |

### §7. Analysis and discussion (≈1.0 page; ≈600 words)

#### §7.1 Why the inversion is below random rather than at chance (≈150 words)

The predictor learns family-specific directions from the training pair distribution; on the unseen family, those directions *extrapolate* in the wrong direction rather than simply being uninformative. Concretely, on `strong_held_out_2`, stable pairs receive higher cosine *distance* than changing pairs, so the AUROC computed against the stable=0/changing=1 convention sits at $1 - \mathrm{AUROC}_{\mathrm{flipped}}$, i.e. systematically below $0.5$. Source for the mechanical account: [EXP-006b §4.4](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md), [EXP-006a §4.4](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md). Pair-score histograms in §4.6 of each report show stable-changing distributions visibly flipped on the unseen family.

**Sample paragraph draft:**

> The below-random orientation of test AUROC under frozen natural-image backbones is not noise but a structured failure: the linear predictor learns a direction $w \in \mathbb{R}^d$ that encodes "stable under `strong + strong_held_out`" as small $\langle w, z_t - z_c \rangle$, and this direction extrapolates with the *opposite* sign when the unseen family `strong_held_out_2` rotates the latent geometry the other way. Because the third family's nuisance is operation-disjoint from the training families, the rotation does not partially align — it anti-aligns. Pair-score histograms (Figure 3, leftmost panel) show this directly: stable pairs on the test split receive higher predicted-target cosine distance than changing pairs, and the gap is large enough (≈0.10) that the AUROC sits in the $0.25$–$0.29$ band rather than at chance.

#### §7.2 Why DermLIP transfers under this scaffold and BiomedCLIP doesn't (≈200 words)

Hypothesis: DermLIP's Derm1M-CLIP pretraining aligns nuisance directions across dermoscopic perturbation families; BiomedCLIP's PMC-15M is too broad to do so. **Caveat:** this hypothesis is not yet distinguishable from "DermLIP saw HAM10000 during pretraining and so its embedding space is well-fit to HAM10000 lesion identity." A direction-structure probe (PCA of stable-pair difference vectors per family) would test the alignment hypothesis directly; EXP-009 would test the contamination alternative. Source: [EXP-007 §4.4](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md), [EXP-008 §4.3](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).

**Sample paragraph draft:**

> The +0.66 AUROC swing from EXP-006b (OpenAI CLIP) to EXP-007 (DermLIP) cannot be explained by architecture, scaffold, or optimiser: all three are held byte-identical. The plausible mechanistic accounts are (i) DermLIP's CLIP loss over $\sim$1M dermatology image-text pairs has organised the embedding space such that a single linear direction captures stable-pair similarity across all three nuisance families uniformly, including `strong_held_out_2`; or (ii) DermLIP saw the HAM10000 raw images during pretraining and its embedding space is well-fit to HAM10000 lesion identity in a way that survives the synthetic post-hoc nuisance. EXP-008 rules out a third candidate — "any medical-image pretraining" — by showing that BiomedCLIP, trained on $\sim$15M PMC-15M figure-caption pairs without HAM10000, lifts test AUROC by only $+0.04$ over web CLIP. EXP-008 cannot, however, choose between (i) and (ii); EXP-009 (a non-HAM10000 dermoscopy SSL pretrain) is the experiment that would.

#### §7.3 Loss vs AUROC (≈120 words)

MSE loss delta does not predict train AUROC across optimisers. EXP-006a Adam reduces train MSE by $3.3\%$ and reaches train AUROC $0.893$; EXP-004 SGD reduces train MSE by $13\%$ and reaches train AUROC $0.900$. The optimisers move the predictor's *function* very differently per unit of L2-loss reduction in 768-d normalised space, so reporting both is essential. Source: [EXP-006a §4.3](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) (cross-run loss-trajectory table); per-run training reports under `outputs/hf-runs/<run-id>/artifacts/reports/jepa_training_report.json`.

#### §7.4 What is not yet claimed (≈130 words)

We do not claim that frozen DermLIP "solves" the proxy in a transfer sense. We claim that frozen DermLIP + linear scaffold reaches AUROC $0.944 \pm 0.003$ on `strong_held_out_2` under the specific evaluation protocol of §3, with HAM10000 contamination unpartitioned. The result is consistent with two hypotheses (dermoscopy-domain transfer; HAM10000 image-level overlap during DermLIP pretraining) and EXP-009 is required to choose between them. The BiomedCLIP partition rules out only the broader "any medical-image pretraining is sufficient" alternative. The contamination caveat is load-bearing for the paper's claim and is restated in §1, §6 (Table 1 footnote), §8 (the EXP-009 design), and §9 (limitation 1).

### §8. Required follow-up work — EXP-009 (≈0.5 page; ≈300 words)

Subsections:

| Subsection | Words | Content |
|---|---:|---|
| §8.1 The partition question | 70 | Define precisely what is unresolved. Two hypotheses, the BiomedCLIP partition only rules out the broader alternative. |
| §8.2 EXP-009 design | 90 | Self-pretrain DINOv2 ViT-B/14 on a non-HAM10000 dermoscopy corpus (ISIC archives minus HAM10000 split, DermNet, BCN20000 non-HAM10000 components, MSK-1/2/3/4 non-HAM10000). Short JEPA-style or MIM objective ($\sim$20 epochs). Run the EXP-004 recipe on top. |
| §8.3 Decision table | 0 (table) | Three outcome bands → three claim revisions. Reproduced verbatim from [EXP-008 §7](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md). |
| §8.4 Why not run before submission | 60 | Corpus-assembly cost (provenance audit on every ISIC component, SSL pretrain compute, pretrain-quality check on a held-out HAM10000 split). The methodology and the nine-run arc deserve community feedback before committing to multi-week effort. |
| §8.5 Feedback being sought | 80 | Numbered list: (1) Is the proxy-task design reasonable? (2) Is three-family disjointness the right granularity? (3) Is the proposed partition right or would a different partition (e.g. dermoscopy-text-only retrieval probe over DermLIP) be more informative? (4) What other follow-ups should matter for the next stage? |

Source pointers:

- EXP-009 design draft: [EXP-008 §7](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md), [EXP-007 §7](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).
- Decision table: [EXP-008 §7](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md).

### §9. Limitations (≈0.5 page; ≈300 words)

1. **Pretraining contamination is unpartitioned.** Central caveat. EXP-009 (§8) addresses it. Source: [EXP-007 §6 #1](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md).
2. **Synthetic nuisance.** `strong`, `strong_held_out`, `strong_held_out_2` are deterministic operations, not real-world acquisition variability. Operation lists in §3.3.
3. **Single-architecture comparison.** All four backbones are ViT-B-class. Other architectures (DINOv2-style self-distillation, transformer-only image backbones, SigLIP) might behave differently.
4. **Single dataset.** HAM10000 only. ISIC archive components and PAD-UFES-20 are unprobed in this work.
5. **HAM10000 is cross-sectional.** No real same-lesion-over-time pairs; the proxy is synthetic throughout.
6. **One scaffold class on the dermoscopy backbone.** Only linear; MLP-on-DermLIP not run.

### §10. Conclusion (≈0.25 page; ≈200 words)

Restate the headline (pretraining-data domain is the load-bearing axis under this protocol, with caveats); restate the non-claim (the underlying thesis is neither validated nor invalidated at this stage); point at EXP-009 and at real longitudinal data as the natural extensions. End with the feedback ask.

Source pointers:

- Headline: [EXP-008 §5.2](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) (updated thesis statement).
- Honest non-claim: [EXP-007 §5.1](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md) (the four "proved / not proved" bullets).

### §11. Contributions (placement: bullet list at end of §1; ≈100 words)

1. **Methodological.** Leakage-controlled HAM10000 longitudinal-proxy task with three disjoint synthetic nuisance families. Released as code + configs + reproduction launchers.
2. **Empirical (negative).** Across two architectures × three predictor scaffolds × two optimisers, frozen natural-image backbones produce below-random inverted test AUROC on the held-out third family.
3. **Empirical (positive but unpartitioned).** Frozen DermLIP reaches test AUROC $0.944 \pm 0.003$ across 5 seeds, $+0.364$ above the strongest baseline. The win cannot yet be attributed to dermoscopy-domain transfer rather than HAM10000 image-level overlap; EXP-009 (§8) is the open partition.
4. **Empirical (negative-as-partition).** Frozen BiomedCLIP — the cleanest publicly available "general medical pretraining without HAM10000" backbone — lifts only $+0.04$ over web CLIP, ruling out "any medical-image pretraining" as sufficient.
5. **Reproducibility.** Every primary-tier run is archived publicly with manifests, embeddings, metrics, model card, and logs at [`abdelstark/derma-jepa-runs/<run_id>`](https://huggingface.co/datasets/abdelstark/derma-jepa-runs). A single launcher script reproduces each run end-to-end on a free A10G via Hugging Face Jobs.
6. **A clearly-scoped open question.** EXP-009's design, decision table, and place in the paper are stated up front rather than treated as future work in passing.

---

## 4. Per-section page and word budget

Total target: 8 pages of body + 2 pages of references = 10 pages (TMLR / arXiv comfortable; MIDL short = pick a 4-page subset).

| Section | Pages | Words | Sub-budget |
|---|---:|---:|---|
| Abstract | 0.25 | 260 | single paragraph |
| §1 Introduction | 1.00 | 600 | 7 blocks per §1 above |
| §2 Related work | 0.75 | 450 | 3 paragraphs × 150 |
| §3 Method | 1.00 | 600 | equations + 5 short subsections |
| §4 Experimental setup | 0.50 | 300 | table + 3 short blocks |
| §5 Experimental sequence | 1.00 | 600 | 9 paragraphs × 60–70 |
| §6 Results | 1.00 | 200 | dominated by figures and tables |
| §7 Analysis & discussion | 1.00 | 600 | 4 subsections per §7 above |
| §8 Required follow-up | 0.50 | 300 | 5 subsections + decision table |
| §9 Limitations | 0.50 | 300 | 6 numbered items |
| §10 Conclusion | 0.25 | 200 | single paragraph |
| **Body total** | **7.75** | **4,410** | |
| References | 1.5–2.0 | — | ~25 entries |
| Appendix (optional) | 4–8 | — | A through I per §7 below |

For a MIDL short paper (4 pages + refs), keep §1, §3 (compressed), §6, §8, §9, §10. Drop §2 to a single sentence and §5 to a 4-row summary table.

---

## 5. Numbers safe to cite (single source of truth)

Pulled from the locked seed-sweep summary and per-run reports. Every number cited in the paper must come from this list. Updates here propagate to the abstract, §1, §5, §6, and the conclusion.

- DermLIP-linear test AUROC: **0.944 ± 0.003** (n = 5 seeds, range 0.939–0.947, 95 % CI[mean] [0.941, 0.946]) — [seed-sweep §3](../experiments/EXP-007-008-seed-sweep-summary.md)
- BiomedCLIP-linear test AUROC: **0.329 ± 0.012** (n = 5 seeds, range 0.312–0.344, 95 % CI[mean] [0.318, 0.339]) — [seed-sweep §3](../experiments/EXP-007-008-seed-sweep-summary.md)
- OpenAI CLIP-linear test AUROC: 0.286 [0.265, 0.310] (n = 1) — [EXP-006b §4.1](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md)
- DINOv2 B/14-linear test AUROC: 0.249 [0.230, 0.270] (n = 1, EXP-004) — [EXP-004 §4.1](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md)
- DINOv2 B/14-MLP-Adam test AUROC: 0.248 [0.228, 0.269] (n = 1, EXP-006a) — [EXP-006a §4.1](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md)
- DINOv2 B/14-MLP-SGD test AUROC: 0.270 [0.249, 0.293] (n = 1, EXP-005) — [EXP-005 §4.1](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md)
- Pixel L2 baseline test AUROC: 0.580 [0.556, 0.606] (deterministic, identical across runs) — any post-EXP-004 report §4.1
- SSIM distance test AUROC: 0.436 [0.411, 0.459]
- Web → general-medical step: **+0.04** AUROC
- General-medical → dermoscopy step: **+0.62** AUROC
- Train → test drop, EXP-007: −0.05
- Train → test drop, EXP-006b: −0.70
- DermLIP raw cosine baseline: 0.109 [0.095, 0.124] (still inverted, predictor lifts +0.84)
- BiomedCLIP raw cosine baseline: 0.047 [0.040, 0.055]
- OpenAI CLIP raw cosine baseline: 0.036 [0.030, 0.043]
- HAM10000 split sizes: $5{,}229 / 1{,}120 / 1{,}121$ unique lesions; $1{,}000 + 1{,}000$ pairs per split.
- Per-run wall-clock: ~85 min on a single A10G (single-backbone runs).

---

## 6. Pre-mortem: what would sink this paper

Five failure modes the draft must defend against. For each, the section that handles it.

| Failure mode | Defended in | How |
|---|---|---|
| Reviewer reads the DermLIP headline as an unconditional claim | §1 (preliminary status block), §6 Table 1 footnote, §7.4, §9 limitation 1 | Caveat front-loaded; "We explicitly do not claim that this swing is dermoscopy-domain transfer" verbatim in abstract |
| Reviewer demands EXP-009 be in the paper | §8 (dedicated section), §1 contributions list (item 6) | EXP-009 design, decision table, and "why not run before submission" in §8 — preempts the "you should have done this" objection by addressing it |
| Reviewer claims the synthetic nuisance is unrealistic | §3.3 (operation lists from real photometric / geometric / sensor effects), §9 limitation 2 | The augmentation suite is justified by reference to real lesion-photography acquisition variation; we never claim the proxy is real-world equivalent |
| Reviewer claims the leakage probe is insufficient | §3.4 (lesion-ID-aware splits, deterministic seed) + appendix audit | Every run reports `data_audit.json` showing zero patient/lesion overlap across splits; cite `outputs/hf-runs/<run-id>/data_audit.json` |
| Reviewer claims the contribution is too narrow ("just one dataset, one architecture class") | §9 limitations 3 + 4, §8 follow-up scope | Acknowledged explicitly; methodology contribution stands independent of the empirical scope |

---

## 7. Reference list (target ≈25 entries; full BibTeX in [`docs/paper/WRITING-PLAN.md` §5](WRITING-PLAN.md))

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

## 8. Appendix structure

- **A. Per-run details.** One subsection per primary-tier run, summarising the experiment report's §1 / §4 / §5 / §6.
- **B. Hyperparameters.** Predictor architecture, optimiser, LR, weight decay, batch size, epochs, identity warm-start; per variant.
- **C. Nuisance-family operation specifications.** Verbatim from §3.3 with the line-numbered source pointers.
- **D. Bootstrap CI protocol.** Per-pair resampling, 1,000 samples, 95 % bands. Source: [`src/derma_jepa/metrics.py:66` `bootstrap_auroc_ci`](../../src/derma_jepa/metrics.py); validation: [`tests/test_metrics.py`](../../tests/test_metrics.py).
- **E. Representation-health checks.** Prediction-norm, dimension-variance, collapse flag definitions and per-run values. Source: [`src/derma_jepa/training.py:479` `_collapse_checks`](../../src/derma_jepa/training.py).
- **F. Compute budget.** Per-run wall-clock breakdown; total GPU-hours across all 9 runs + seed sweep.
- **G. Reproducibility checklist.** Following NeurIPS / MLRC standard items, mapped to repository assets.
- **H. Pretraining contamination analysis.** Detailed argument for why HAM10000 is in Derm1M and the boundaries of what EXP-007 / EXP-008 can and cannot conclude. Load-bearing for the paper.
- **I. EXP-009 design (planned).** Corpus assembly plan, SSL pretrain protocol, decision table mapping outcome to claim revision.

---

## 9. Submission targets

In rough order of fit, prioritising venues that accept honest preliminary reports:

| Venue | Track / format | Fit | Notes |
|---|---|---|---|
| arXiv preprint | non-archival | always first | Locks priority date. |
| TMLR | full paper, rolling | high | Empirical / methodology track, no page limit, accepts honest preliminary work. Recommended primary venue. |
| MIDL 2026 | short paper (4 pp + refs) | high | Short-paper format suits the preliminary-evidence framing. |
| ML4H 2026 (NeurIPS workshop) | full or extended-abstract | high | Health-ML community feedback is exactly what is wanted at this stage. |
| MICCAI 2026 (workshop, e.g. ISIC, MILLanD) | workshop paper | medium-high | Full MICCAI track is harder without EXP-009. |
| MIDL 2026 (full) | full paper | medium | Defer to v2 if EXP-009 lands. |
| NeurIPS 2026 Datasets & Benchmarks | full paper | medium | Better as a v2 venue after EXP-009. |
| Full NeurIPS / ICLR | full paper | low for this paper | Wait until EXP-009 closes the partition. |

Concrete plan: arXiv preprint now, TMLR or ML4H workshop submission within 4 weeks, then a v2 with EXP-009 results for a MIDL or MICCAI full-paper submission.

---

## 10. Open issues for the writing pass

- Whether the abstract should lead with the negative-result-arc framing or with the DermLIP-specific positive headline. Current outline leads with the gradient and front-loads the contamination caveat. Both framings are honest; reviewer preferences differ.
- Whether to include the BiomedCLIP partition in §5 alongside DermLIP or to give it a dedicated subsection. Outline currently uses dedicated subsections (§5.7 and §5.8).
- Whether to run a direction-structure probe (§7.2) before submission. It would let the family-alignment hypothesis become a result rather than a hypothesis. Compute-cheap; recommend yes if time permits.
- How much of the appendix to ship in v1. For a preliminary preprint, all nine appendices in §8 are useful; for a workshop short paper, B / D / G / H are essential and the rest can move to a supplement repo.

---

## 11. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-28 | Abdelhamid Bakhta | Initial outline. |
| 2026-04-28 | Abdelhamid Bakhta | Reframed as preliminary release without EXP-009. |
| 2026-04-28 | Abdelhamid Bakhta | First deepening pass: per-section source pointers and writing tasks. |
| 2026-04-28 | Abdelhamid Bakhta | Second deepening pass: code-grounded source pointers verified at file:line · symbol; LaTeX equations for the Method section drafted; concrete deterministic-augmentation operation lists for the three nuisance families lifted from `src/derma_jepa/public_data.py`; per-section page and word budget tabulated; sample paragraph drafts inlined for §7.1 and §7.2; pre-mortem table added enumerating five failure modes and the section that defends each; companion document [`docs/paper/REVIEWER-QA.md`](REVIEWER-QA.md) added for anticipated reviewer questions. |
