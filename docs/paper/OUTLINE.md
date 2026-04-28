# DermaJEPA paper outline (v1)

**Status:** Draft outline. Numbers and arguments below trace to per-run reports under `docs/experiments/`. EXP-009 is unblocked but unrun; outline notes both the with-EXP-009 and without-EXP-009 framings.

---

## 1. Title

Three working titles, ranked by directness:

1. **Frozen-backbone JEPA-style probes invert below random on a held-out nuisance family — until the backbone is dermoscopy-pretrained.** A nine-experiment study on HAM10000.
2. *When frozen-backbone JEPA generalises and when it doesn't: a leakage-controlled study on HAM10000 dermoscopic images.*
3. *Pretraining-data domain is the load-bearing axis for frozen-backbone JEPA on dermoscopy: evidence and a contamination caveat.*

Final title contingent on EXP-009 outcome. If EXP-009 reproduces ≈0.94 on a non-HAM10000 dermoscopy SSL pretrain, title (1) stands as written. If EXP-009 lands at 0.30–0.50, the title narrows to "frozen-backbone JEPA succeeds when the backbone has seen the eval images."

---

## 2. Abstract (≈250 words)

Self-supervised representations promise robustness to nuisance variation; whether that robustness transfers across novel nuisance families is rarely tested. We probe this on a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families — illumination, geometric, and colour-perturbation operations partitioned along the deterministic-augmentation axis — training a JEPA-style latent predictor on stable pairs from two of three families and evaluating on the third unseen family. Across two frozen natural-image backbones (DINOv2 ViT-B/14, OpenAI CLIP ViT-B/16), three predictor scaffolds (linear, underfit MLP, fit MLP under Adam), and two optimisers (SGD, Adam), test AUROC on the held-out family is 0.25–0.29 — below random and below cheap baselines (pixel L2, SSIM, raw embedding cosine). A frozen general-medical backbone (BiomedCLIP, PMC-15M) lifts test AUROC by only +0.04 to 0.329 ± 0.012 across 5 seeds. A frozen dermoscopy-pretrained backbone (DermLIP, Derm1M) lifts test AUROC to 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline and the only above-random result on the held-out family across 9 runs. Holding architecture (ViT-B/16) and predictor (linear) constant, varying only the pretraining-data domain produces a +0.66 AUROC swing localised to the dermoscopy-specific step. The win comes with an explicit caveat: DermLIP's pretraining corpus, Derm1M, almost certainly includes HAM10000, and our experiments cannot separate dermoscopy-domain transfer from HAM10000 image-level overlap. We release the experiment harness, the configurations, and a public run archive for every primary-tier run, including a 5-seed sweep that locks the DermLIP and BiomedCLIP point estimates.

---

## 3. Section structure

### 1. Introduction (≈1.0 page)

- **Motivation.** Lesion monitoring is fundamentally a change-detection problem under heavy nuisance variation (illumination, angle, hair, framing). Self-supervised representations are pitched as nuisance-invariant; the question we test is whether that invariance generalises across *novel* nuisance families a model never saw during training.
- **Setting.** Public HAM10000 dermoscopy data is cross-sectional, so we construct a leakage-controlled longitudinal proxy: lesion-ID-aware splits, synthetic stable-pair augmentations applied post-split, changing-pair construction matched on diagnosis and anatomical site.
- **Probe.** A JEPA-style latent predictor (linear or 2-layer identity-residual MLP) over a frozen vision backbone, trained to minimise L2 between predicted and observed target latents on stable pairs.
- **One-axis-at-a-time ablation.** Across nine experiments we vary predictor class, optimiser, and backbone independently while holding everything else constant.
- **Contributions** (see §10).
- **Headline.** Figure 1: the cross-backbone gradient. Frozen natural-image backbones invert below random; general-medical pretraining barely lifts; dermoscopy-specific pretraining is at 0.944 ± 0.003.
- **Caveat.** Section 1 must front-load the HAM10000 contamination caveat for DermLIP. The honest paper is "we observe X with caveat Y, and EXP-009 (or its conclusion if run) is the experiment that bounds Y."

### 2. Related work (≈0.75 page)

Three threads, each one paragraph:

- **JEPA-style and self-supervised image representations.** [LeCun 2022], [Assran et al. 2023] (I-JEPA), [Caron et al. 2021] (DINO), [Oquab et al. 2024] (DINOv2), [Radford et al. 2021] (CLIP), [Cherti et al. 2023] (OpenCLIP). Position our probe as "the frozen-backbone, single-step, stable-pair subset of the JEPA family applied as an evaluation probe rather than a pretraining objective."
- **Domain-specific medical foundation models.** BiomedCLIP [Zhang et al. 2023], DermLIP / PanDerm [Yan et al. 2025], MedCLIP, MONET. We use BiomedCLIP and DermLIP as third-party frozen weights.
- **Distribution-shift evaluation in dermoscopy / medical imaging.** ISIC challenge series, RETFound and similar foundation-model evaluations, Wilds-style covariate-shift benchmarks. Our contribution is the *deterministic, leakage-controlled, three-family-disjoint nuisance design* that exposes a sharp failure mode previously not characterised at this granularity.

### 3. Method (≈1.0 page)

- **§3.1 Notation and probe.** For each frozen backbone $f$ and image $x$, $z = f(x) / \lVert f(x) \rVert_2 \in \mathbb{R}^d$. Stable pairs $(x_c, x_t)$: same lesion under synthetic nuisance. Changing pairs: distinct lesions matched on diagnosis and anatomical site. Predictor $g_\theta : \mathbb{R}^d \to \mathbb{R}^d$ trained to minimise $\lVert g_\theta(z_c) - z_t \rVert_2^2$ on stable training pairs. Evaluation: cosine distance between $g_\theta(z_c)$ and $z_t$.
- **§3.2 Predictor variants.**
  - Linear with identity warm-start: $g_\theta(z) = (I + W) z + b$, $W$ initialised small Gaussian, $b = 0$. Soft `weight − I` regulariser.
  - 2-layer identity-residual MLP: $g_\theta(z) = z + W_2 \,\text{ReLU}(W_1 z + b_1) + b_2$, $W_2$ zero-initialised so $g_\theta = \text{identity}$ at step 0.
- **§3.3 Three nuisance families.** Specify the deterministic operations in `strong`, `strong_held_out`, `strong_held_out_2`. The disjointness rule is operation-level disjointness, not parameter-level disjointness.
- **§3.4 Splits and pair construction.** Lesion-ID-aware splits (5,229 / 1,120 / 1,121). 1,000 stable + 1,000 changing pairs per split. Strict same-diagnosis-site changing-pair policy.
- **§3.5 Evaluation protocol.** AUROC primary; 1,000-sample bootstrap 95 % CIs over groups; AUPRC, EER, FPR-at-fixed-TPR(0.8), and three representation-health checks (prediction-norm, dimension-variance, collapse flag) reported on every run.

### 4. Experimental setup (≈0.5 page)

- **§4.1 Backbones.** Cite each weight set with HF Hub path and primary citation (table replicated from README §Backbones).
- **§4.2 Training.** SGD vs Adam, hyperparameters, 200 epochs at batch 128, identical across experiments except for the explicitly varied axis.
- **§4.3 Compute.** Single A10G 24 GB on Hugging Face Jobs; ~85 min wall-clock per run end-to-end. `scripts/hf_jobs_constraints.txt` pins all dependency versions.
- **§4.4 Code, configurations, and run archive.** GitHub repo + HF dataset link.

### 5. Experimental sequence (≈1.0 page)

The arc as a falsification ladder. One paragraph per stage, headline number per stage:

- **§5.1 Sanity (EXP-001).** Trivial proxy: every dense baseline at AUROC ≈ 1.0. The proxy needs hardening.
- **§5.2 Hardened proxy, matched eval (EXP-002).** JEPA wins +0.27 AUROC over the strongest baseline (DINOv2-S cosine = 0.652). Real signal when test nuisance matches training.
- **§5.3 One-family-held-out (EXP-003).** Win collapses to a loss (−0.28 vs SSIM = 0.961). In-distribution invariance does not transfer to one disjoint nuisance family.
- **§5.4 Three-family training, third-family eval (EXP-004).** Predictor inverts to 0.249 AUROC, −0.33 below pixel L2 = 0.580. Below-random *inverted* test ranking.
- **§5.5 Scaffold ablation (EXP-005, EXP-006a).** MLP under SGD underfits (train AUROC 0.572, test 0.270); MLP under Adam fits (train 0.893) and lands at the same test AUROC 0.248. Scaffold-capacity hypothesis falsified.
- **§5.6 Backbone ablation, natural-image (EXP-006b).** OpenAI CLIP ViT-B/16 with the EXP-004 recipe lands at test 0.286 — within bootstrap noise of DINOv2's 0.249. Inversion is not DINOv2-specific.
- **§5.7 Pretraining-domain ablation (EXP-007).** DermLIP (CLIP-trained on Derm1M) under the same recipe lands at test 0.945. Train-test drop collapses from −0.70 to −0.05.
- **§5.8 Pretraining-domain partition (EXP-008).** BiomedCLIP (PMC-15M, no HAM10000) lands at 0.325. Lift from web to general-medical is +0.04; lift from general-medical to dermoscopy is +0.62.
- **§5.9 Seed robustness.** 5-seed sweep on EXP-007 and EXP-008. Test AUROC mean ± std: DermLIP 0.944 ± 0.003, BiomedCLIP 0.329 ± 0.012. Both headlines seed-stable.

### 6. Results (≈1.0 page, dominated by tables and figures)

- **Table 1.** Cross-run results (the README headline table, expanded with bootstrap CIs and seed std).
- **Figure 1.** Test-AUROC vs pretraining-data domain (web < general-medical < dermoscopy), showing the 15× non-uniform gradient.
- **Figure 2.** Train-vs-test AUROC scatter across runs 4–8. Train AUROC spans 0.57–0.99; test AUROC spans 0.25–0.95. Decoupled along the backbone axis.
- **Figure 3.** Stable / changing pair-score histograms on test split for EXP-006b (OpenAI CLIP), EXP-008 (BiomedCLIP), EXP-007 (DermLIP). Three-panel side-by-side showing the geometric story: web and general-medical backbones produce inverted distributions; DermLIP cleanly separates.
- **Table 2.** Predictor-class × optimiser ablation on DINOv2 ViT-B/14 (EXP-004 / EXP-005 / EXP-006a). Train AUROC 0.572–0.900; test AUROC 0.248–0.270.
- **Table 3.** Seed sweep: per-seed train / val / test AUROC for EXP-007 and EXP-008 plus across-seed mean / std / 95 % CI[mean].
- **Representation-health table.** Prediction-norm and dimension-variance per run (collapse always False).

### 7. Analysis and discussion (≈1.0 page)

- **§7.1 Why the inversion is below random rather than at chance.** The predictor learns family-specific directions on the training pair distribution; on the unseen family those directions extrapolate in the opposite direction, producing systematically wrong rankings rather than uncorrelated rankings.
- **§7.2 Why DermLIP transfers and BiomedCLIP doesn't.** Hypothesis: DermLIP's Derm1M-CLIP pretraining aligns nuisance directions across dermoscopic perturbation families; BiomedCLIP's PMC-15M is too broad to do so. Provisional; a direction-structure probe (PCA of stable-pair difference vectors per family) would test this directly.
- **§7.3 Loss vs AUROC.** Train MSE delta does not predict train AUROC across optimisers (EXP-006a Adam reduces MSE 4× less than EXP-004 SGD and reaches equivalent train AUROC). Reporting both is essential.
- **§7.4 Pretraining contamination.** Derm1M is not published with a per-source breakdown; given HAM10000's prominence as a public dermoscopy corpus, DermLIP almost certainly saw HAM10000 raw images. The synthetic nuisance augmentations the labels depend on are post-hoc and unpublished, so the labels do not leak; the *images* do. EXP-009 (a non-HAM10000 dermoscopy SSL pretrain) is the partition experiment.

### 8. Limitations (≈0.5 page)

1. **Pretraining contamination unpartitioned.** EXP-009 pending or completed (insert outcome).
2. **Synthetic nuisance.** `strong`, `strong_held_out`, `strong_held_out_2` are deterministic operations, not real-world acquisition variability.
3. **Single-architecture comparison.** All four backbones are ViT-B-class. Other architectures (DINOv2-style self-distillation, transformer-only image backbones, SigLIP) might behave differently.
4. **Single dataset.** HAM10000 is one dermoscopy benchmark; ISIC archive components and PAD-UFES-20 are unprobed in this work.
5. **HAM10000 is cross-sectional.** No real same-lesion-over-time pairs; the proxy is synthetic throughout.
6. **One scaffold class on the dermoscopy backbone.** Only linear + Adam-MLP ablations on DINOv2; we did not run MLP-on-DermLIP.

### 9. Conclusion (≈0.25 page)

Restate the headline: pretraining-data domain is the load-bearing axis. Rephrase contributions in two-sentence form. Point at EXP-009 (or its outcome) and at real longitudinal data as the natural extensions.

### 10. Contributions (target placement: bullet list at end of §1)

1. **Methodological.** A leakage-controlled HAM10000 longitudinal-proxy task with three disjoint synthetic nuisance families designed to isolate "fits the seen distribution" from "generalises out of distribution," released as code + configs + reproduction launchers.
2. **Empirical (negative).** Across two architectures × three predictor scaffolds × two optimisers, frozen natural-image backbones produce below-random *inverted* test AUROC on the held-out third family. The failure is robust along scaffold and optimiser axes.
3. **Empirical (positive).** Frozen DermLIP under an identical recipe reaches test AUROC 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline. The win is localised to the dermoscopy-specific pretraining step, with a +0.04 contribution from "any medical pretraining" (BiomedCLIP, 5 seeds).
4. **Reproducibility.** Every primary-tier run is archived publicly with manifests, embeddings, metrics, model card, and logs at `abdelstark/derma-jepa-runs/<run_id>`. A single launcher script reproduces each run end-to-end on a free A10G via Hugging Face Jobs.
5. **Caveat as contribution.** The HAM10000 contamination caveat on the DermLIP headline is front-loaded, the BiomedCLIP partition experiment is reported, and EXP-009 is scoped with a decision table mapping its outcome to a paper-headline revision.

---

## 4. Key figures and tables (master list)

| Asset | Source | Where it appears |
|---|---|---|
| Table 1: Cross-run results | EXP-008 §9.4 + seed-sweep summary §4 | §1 (headline), §6 (full) |
| Figure 1: Test AUROC vs pretraining domain | New panel from cross-run table | §6 |
| Figure 2: Train-vs-test AUROC scatter | New plot from per-run `metrics.json` | §6 |
| Figure 3: 3-panel JEPA score histograms | `artifacts/plots/jepa_score_histogram.png` from EXP-006b / EXP-007 / EXP-008 | §6 |
| Table 2: Predictor × optimiser ablation | EXP-004 / EXP-005 / EXP-006a §4.1 each | §6 |
| Table 3: Seed sweep | Seed-sweep summary §3 | §6 |
| Table 4: Representation health | per-run §4.5 | Appendix |
| Pipeline diagram | README Mermaid (§ Pipeline) | §3 or §4 |
| Three-family experimental design diagram | README Mermaid (§ Experimental design) | §3 |
| Predictor-objective diagram | README Mermaid (§ Method) | §3.1 |

For paper rendering, the README Mermaid sources should be re-rendered as TikZ or vector PDF before submission. Mermaid is fine for the GitHub README but not for camera-ready.

---

## 5. Numbers safe to cite (single source of truth)

Pulled from the locked seed-sweep summary and per-run reports. Do not retype from the abstract; cite from this list.

- DermLIP-linear test AUROC: **0.944 ± 0.003** (n = 5 seeds, range 0.939–0.947, 95 % CI[mean] [0.941, 0.946])
- BiomedCLIP-linear test AUROC: **0.329 ± 0.012** (n = 5 seeds, range 0.312–0.344, 95 % CI[mean] [0.318, 0.339])
- OpenAI CLIP-linear test AUROC: 0.286 [0.265, 0.310] (n = 1)
- DINOv2 B/14-linear test AUROC: 0.249 [0.230, 0.270] (n = 1, EXP-004)
- DINOv2 B/14-MLP-Adam test AUROC: 0.248 [0.228, 0.269] (n = 1, EXP-006a)
- Pixel L2 baseline test AUROC: 0.580 [0.556, 0.606] (deterministic, identical across runs)
- SSIM distance test AUROC: 0.436 [0.411, 0.459]
- Web → general-medical step: **+0.04** AUROC
- General-medical → dermoscopy step: **+0.62** AUROC
- Train → test drop, EXP-007: −0.05
- Train → test drop, EXP-006b: −0.70
- DermLIP raw cosine baseline: 0.109 [0.095, 0.124] (still inverted, predictor lifts +0.84)
- BiomedCLIP raw cosine baseline: 0.047 [0.040, 0.055]
- OpenAI CLIP raw cosine baseline: 0.036 [0.030, 0.043]

---

## 6. Reference list (target ≈30 entries)

Already in `README.md` §References. Expand to include:

- Caron, M. et al. *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021. arXiv:2104.14294.
- He, K. et al. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. arXiv:2111.06377.
- Bardes, A., Ponce, J., LeCun, Y. *VICReg*. ICLR 2022. arXiv:2105.04906.
- Tolstikhin, I. et al. *MLP-Mixer*. NeurIPS 2021 (cite if reviewers ask about scaffold choice).
- Zhang et al. *Understanding Deep Learning Requires Rethinking Generalization*. ICLR 2017 (the train-vs-test decoupling reference).
- Gulrajani, I. and Lopez-Paz, D. *In Search of Lost Domain Generalization*. ICLR 2021. arXiv:2007.01434 (the natural reference for distribution-shift evaluation methodology).
- Wilds: Koh, P. W. et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML 2021. arXiv:2012.07421.
- ISIC: Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection: A Challenge at ISIC 2017*. arXiv:1710.05006 (and the 2018 / 2019 follow-ups).
- BCN20000: Combalia, M. et al. *BCN20000: Dermoscopic Lesions in the Wild*. arXiv:1908.02288 (cited for EXP-009 corpus design).
- All references currently listed in the README: I-JEPA, OpenCLIP, JEPA position paper, DINOv2, CLIP, HAM10000, DermLIP/Derm1M, BiomedCLIP.

---

## 7. Appendix structure

- **A. Per-run details.** One subsection per primary-tier run, summarising the experiment report's §1 / §4 / §5 / §6.
- **B. Hyperparameters.** Predictor architecture, optimiser, LR, weight decay, batch size, epochs, identity warm-start; for each variant.
- **C. Nuisance-family operation specifications.** Deterministic operation lists for `strong`, `strong_held_out`, `strong_held_out_2`. Exact augmentation-suite source code reference.
- **D. Bootstrap CI protocol.** Per-group resampling, 1,000 samples, 95 % bands.
- **E. Representation-health checks.** Prediction-norm, dimension-variance, collapse flag definitions and per-run values.
- **F. Compute budget.** Per-run wall-clock breakdown; total GPU-hours across all 9 runs + seed sweep.
- **G. Reproducibility checklist.** Following NeurIPS / MLRC standard items, mapped to repository assets.
- **H. Pretraining contamination analysis.** Detailed argument for why HAM10000 is in Derm1M and the boundaries of what EXP-007 / EXP-008 can and cannot conclude.
- **I. EXP-009 design (if not run by submission).** Or **EXP-009 results (if run).** Either way: corpus assembly, SSL pretrain protocol, decision table mapping outcome to claim revision.

---

## 8. Submission targets

In rough order of fit:

| Venue | Track / format | Fit | Notes |
|---|---|---|---|
| MICCAI 2026 | full paper (8–10 pages) | high | Medical-imaging audience; will press hard on contamination, so EXP-009 is recommended before submission. |
| MIDL 2026 | full paper or short paper | high | Smaller, more methodology-tolerant. Short-paper format may be a good fit. |
| NeurIPS 2026 Datasets & Benchmarks | full paper | medium-high | Methodological contribution and run archive fit the track; medical-domain reviewers vary. |
| TMLR | full paper, rolling | high | Empirical / methodology track. No page limit pressure. Probably the best venue for a paper that wants to spend two pages on the contamination caveat. |
| arXiv preprint | non-archival | always | Recommended ahead of any venue submission to lock priority date. |

Short-paper / preprint format (≈6 pages) would compress §3 and §4 and skip the appendix; long format (≈10 pages + appendix) keeps everything.

---

## 9. Open issues for the writing pass

- Whether to lead the abstract with the negative-result-arc framing or with the DermLIP-specific positive headline. Current outline leads with the gradient. Reviewer preferences differ; both framings are honest.
- Whether to include EXP-009 results in the same paper or as a separate short follow-up. Recommendation: same paper if EXP-009 confirms ≥ 0.85; separate paper if EXP-009 lands in the partial-transfer band (0.50–0.80) and warrants its own analysis.
- How much of §7 (mechanism analysis) to cut for venues with strict page limits. The PCA / direction-structure probe is currently a hypothesis paragraph; running it would let it become a result.
- Whether the proxy-task design is the headline contribution or a methodology vehicle. Outline currently treats it as a vehicle; if reviewers respond well, future work could promote it to a benchmark on its own.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-28 | AbdelStark + Claude Code | Initial outline. Numbers aligned with the locked seed-sweep summary. EXP-009 framed as the open partition experiment. |
