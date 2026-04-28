# DermaJEPA paper outline (v1, preliminary release)

**Status:** Working outline for a preliminary preprint. The paper releases the nine-experiment arc, the proxy-task design, the run archive, and the locked seed-sweep numbers as a methodology-and-early-evidence contribution. EXP-009 (the dermoscopy-domain-transfer vs HAM10000-contamination partition) is **not** in scope for this paper; it is reported as required follow-up work with an explicit decision table. Numbers and arguments below trace to per-run reports under [`docs/experiments/`](../experiments/README.md).

---

## 0. Framing

This paper is a *preliminary report*, not a closed result. We share the arc at this stage because:

1. The methodology — a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families and a `train-on-two / evaluate-on-third` protocol — is novel enough that it should be put in front of reviewers before further compute is spent extending it.
2. The nine-run sequence already characterises a clean failure mode of frozen natural-image and frozen general-medical backbones (test AUROC inverts below random across two architectures, three predictor scaffolds, two optimisers).
3. The single positive result observed (frozen DermLIP at 0.944 ± 0.003 across 5 seeds) **cannot be attributed to dermoscopy-domain transfer with the experiments in this paper alone**. DermLIP's pretraining corpus, Derm1M, almost certainly contains HAM10000 raw images, and the BiomedCLIP partition (EXP-008) only rules out the broader "any medical-image pretraining" alternative — not HAM10000 image-level overlap specifically.
4. EXP-009 (a self-pretrained DINOv2 on a non-HAM10000 dermoscopy corpus) is the experiment that would partition dermoscopy-domain transfer from contamination. Building that corpus and running the SSL pretrain is a multi-week engineering task. We want feedback on the methodology, the proxy design, and whether the partition we have proposed is the right one before committing to it.

The paper is therefore framed as: methodology + nine-experiment characterisation + one acknowledged-non-representative positive result + scoped follow-up. We treat this as the honest preprint state.

---

## 1. Title

Three working titles, all explicit about the preliminary status:

1. **Frozen-backbone JEPA-style probes on HAM10000: preliminary results from a nine-experiment ablation, with an unpartitioned contamination caveat.**
2. *Toward a clean test of frozen-backbone JEPA generalisation on dermoscopy: methodology, early evidence, and an open partition experiment.*
3. *When does frozen-backbone JEPA generalise on dermoscopy? A preliminary nine-experiment study and a planned partition.*

Final title locks at submission. The "preliminary" qualifier stays in the title regardless.

---

## 2. Abstract (≈260 words)

Self-supervised representations promise robustness to nuisance variation; whether that robustness transfers across novel nuisance families is rarely tested. We probe this on a leakage-controlled HAM10000 longitudinal proxy with three disjoint synthetic nuisance families — partitioned along the deterministic-augmentation axis — training a JEPA-style latent predictor on stable pairs from two of three families and evaluating on the third unseen family. We report a preliminary nine-experiment arc. Across two frozen natural-image backbones (DINOv2 ViT-B/14, OpenAI CLIP ViT-B/16), three predictor scaffolds (linear, underfit MLP, fit MLP under Adam), and two optimisers (SGD, Adam), test AUROC on the held-out family is 0.25–0.29 — below random and below cheap baselines (pixel L2, SSIM, raw embedding cosine). A frozen general-medical backbone (BiomedCLIP, PMC-15M, no HAM10000) lifts test AUROC by only +0.04 to 0.329 ± 0.012 across 5 seeds. A frozen dermoscopy-pretrained backbone (DermLIP, Derm1M) lifts test AUROC to 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline and the only above-random result on the held-out family across nine runs. Holding architecture (ViT-B/16) and predictor (linear) constant, varying only the pretraining-data domain produces a +0.66 AUROC swing localised to the dermoscopy-specific step. We **explicitly do not claim that this swing is dermoscopy-domain transfer**: DermLIP's Derm1M corpus almost certainly contains HAM10000, and the experiments reported here cannot separate dermoscopy-domain transfer from HAM10000 image-level overlap. We release the methodology, the configurations, the run archive, and a planned partition experiment (a non-HAM10000 dermoscopy SSL pretrain) as required follow-up. The paper's contribution is the proxy-task design, the failure characterisation, and a clearly scoped open question — not a conclusion about whether frozen-backbone JEPA generalises on dermoscopy.

---

## 3. Section structure

### 1. Introduction (≈1.0 page)

- **Motivation.** Lesion monitoring is fundamentally a change-detection problem under heavy nuisance variation (illumination, angle, hair, framing). Self-supervised representations are pitched as nuisance-invariant; the question we test is whether that invariance generalises across novel nuisance families a model never saw during training.
- **Setting.** Public HAM10000 dermoscopy data is cross-sectional, so we construct a leakage-controlled longitudinal proxy: lesion-ID-aware splits, synthetic stable-pair augmentations applied post-split, changing-pair construction matched on diagnosis and anatomical site.
- **Probe.** A JEPA-style latent predictor (linear or 2-layer identity-residual MLP) over a frozen vision backbone, trained to minimise L2 between predicted and observed target latents on stable pairs.
- **One-axis-at-a-time ablation.** Across nine experiments we vary predictor class, optimiser, and backbone independently while holding everything else constant.
- **Preliminary status.** Explicit paragraph: this is an early-stage report; the EXP-007 positive result is unpartitioned with respect to HAM10000 contamination; EXP-009 is required to close the question; we share now to validate the methodology and to invite feedback on the partition design and on what a stricter test would look like.
- **Contributions** (see §10): all framed as methodology + characterisation + open question, not "we solve X."
- **Headline figure.** Figure 1: the cross-backbone gradient (web < general-medical < dermoscopy on test AUROC, with the contamination caveat annotated on the dermoscopy bar).

### 2. Related work (≈0.75 page)

Three threads, each one paragraph:

- **JEPA-style and self-supervised image representations.** [LeCun 2022], [Assran et al. 2023] (I-JEPA), [Caron et al. 2021] (DINO), [Oquab et al. 2024] (DINOv2), [Radford et al. 2021] (CLIP), [Cherti et al. 2023] (OpenCLIP). Position our probe as "the frozen-backbone, single-step, stable-pair subset of the JEPA family applied as an evaluation probe rather than a pretraining objective."
- **Domain-specific medical foundation models.** BiomedCLIP [Zhang et al. 2023], DermLIP / PanDerm [Yan et al. 2025], MedCLIP, MONET. We use BiomedCLIP and DermLIP as third-party frozen weights.
- **Distribution-shift evaluation in dermoscopy / medical imaging.** ISIC challenge series, RETFound and similar foundation-model evaluations, WILDS-style covariate-shift benchmarks [Koh et al. 2021], in-search-of-lost-DG [Gulrajani-Lopez-Paz 2021]. Our contribution is the deterministic, leakage-controlled, three-family-disjoint nuisance design that exposes a sharp failure mode previously not characterised at this granularity.

### 3. Method (≈1.0 page)

- **§3.1 Notation and probe.** For each frozen backbone $f$ and image $x$, $z = f(x) / \lVert f(x) \rVert_2 \in \mathbb{R}^d$. Stable pairs $(x_c, x_t)$: same lesion under synthetic nuisance. Changing pairs: distinct lesions matched on diagnosis and anatomical site. Predictor $g_\theta : \mathbb{R}^d \to \mathbb{R}^d$ trained to minimise $\lVert g_\theta(z_c) - z_t \rVert_2^2$ on stable training pairs. Evaluation: cosine distance between $g_\theta(z_c)$ and $z_t$.
- **§3.2 Predictor variants.**
  - Linear with identity warm-start: $g_\theta(z) = (I + W) z + b$, $W$ initialised small Gaussian, $b = 0$. Soft `weight − I` regulariser.
  - 2-layer identity-residual MLP: $g_\theta(z) = z + W_2 \,\text{ReLU}(W_1 z + b_1) + b_2$, $W_2$ zero-initialised so $g_\theta = \text{identity}$ at step 0.
- **§3.3 Three nuisance families.** Specify the deterministic operations in `strong`, `strong_held_out`, `strong_held_out_2`. The disjointness rule is operation-level disjointness, not parameter-level disjointness.
- **§3.4 Splits and pair construction.** Lesion-ID-aware splits (5,229 / 1,120 / 1,121). 1,000 stable + 1,000 changing pairs per split. Strict same-diagnosis-site changing-pair policy.
- **§3.5 Evaluation protocol.** AUROC primary; 1,000-sample bootstrap 95 % CIs over groups; AUPRC, EER, FPR-at-fixed-TPR(0.8), and three representation-health checks (prediction-norm, dimension-variance, collapse flag) on every run.

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
- **§5.4 Three-family training, third-family eval (EXP-004).** Predictor inverts to 0.249 AUROC, −0.33 below pixel L2 = 0.580. Below-random inverted test ranking.
- **§5.5 Scaffold ablation (EXP-005, EXP-006a).** MLP under SGD underfits (train AUROC 0.572, test 0.270); MLP under Adam fits (train 0.893) and lands at the same test AUROC 0.248. Scaffold-capacity hypothesis falsified.
- **§5.6 Backbone ablation, natural-image (EXP-006b).** OpenAI CLIP ViT-B/16 with the EXP-004 recipe lands at test 0.286 — within bootstrap noise of DINOv2's 0.249. Inversion is not DINOv2-specific.
- **§5.7 Pretraining-domain ablation (EXP-007).** DermLIP (CLIP-trained on Derm1M) under the same recipe lands at test 0.945. Train-test drop collapses from −0.70 to −0.05. We immediately note that this result is unpartitioned with respect to HAM10000 contamination and proceed to EXP-008.
- **§5.8 Pretraining-domain partition (EXP-008).** BiomedCLIP (PMC-15M, no HAM10000) lands at 0.325. Lift from web to general-medical is +0.04; lift from general-medical to dermoscopy is +0.62. EXP-008 rules out "any medical-image pretraining" but does not separate "dermoscopy-domain transfer" from "HAM10000 image-level overlap." That partition is the open question for EXP-009 (§8 of this paper).
- **§5.9 Seed robustness.** 5-seed sweep on EXP-007 and EXP-008. Test AUROC mean ± std: DermLIP 0.944 ± 0.003, BiomedCLIP 0.329 ± 0.012. Both headlines seed-stable.

### 6. Results (≈1.0 page, dominated by tables and figures)

- **Table 1.** Cross-run results (the README headline table, expanded with bootstrap CIs and seed std). The DermLIP row carries an explicit "contamination caveat: HAM10000 likely in pretrain" annotation.
- **Figure 1.** Test-AUROC vs pretraining-data domain (web < general-medical < dermoscopy), showing the 15× non-uniform gradient. The dermoscopy bar is annotated with the contamination caveat.
- **Figure 2.** Train-vs-test AUROC scatter across runs 4–8. Train AUROC spans 0.57–0.99; test AUROC spans 0.25–0.95. Decoupled along the backbone axis.
- **Figure 3.** Stable / changing pair-score histograms on test split for EXP-006b (OpenAI CLIP), EXP-008 (BiomedCLIP), EXP-007 (DermLIP). Three-panel side-by-side showing the geometric story: web and general-medical backbones produce inverted distributions; DermLIP cleanly separates.
- **Table 2.** Predictor-class × optimiser ablation on DINOv2 ViT-B/14 (EXP-004 / EXP-005 / EXP-006a). Train AUROC 0.572–0.900; test AUROC 0.248–0.270.
- **Table 3.** Seed sweep: per-seed train / val / test AUROC for EXP-007 and EXP-008 plus across-seed mean / std / 95 % CI[mean].
- **Representation-health table.** Prediction-norm and dimension-variance per run (collapse always False). Paper-appendix or supplement.

### 7. Analysis and discussion (≈1.0 page)

- **§7.1 Why the inversion is below random rather than at chance.** The predictor learns family-specific directions on the training pair distribution; on the unseen family those directions extrapolate in the opposite direction, producing systematically wrong rankings rather than uncorrelated rankings.
- **§7.2 Why DermLIP transfers under this scaffold and BiomedCLIP doesn't (with explicit caveat).** Hypothesis: DermLIP's Derm1M-CLIP pretraining aligns nuisance directions across dermoscopic perturbation families; BiomedCLIP's PMC-15M is too broad to do so. **Caveat:** this hypothesis is not yet distinguishable from "DermLIP saw HAM10000 during pretraining and so its embedding space is well-fit to HAM10000 lesion identity." A direction-structure probe (PCA of stable-pair difference vectors per family) would test the family-alignment hypothesis directly; EXP-009 would test the contamination alternative.
- **§7.3 Loss vs AUROC.** Train MSE delta does not predict train AUROC across optimisers (EXP-006a Adam reduces MSE 4× less than EXP-004 SGD and reaches equivalent train AUROC). Reporting both is essential.
- **§7.4 What we do not yet claim.** We do not claim that frozen DermLIP "solves" the proxy in a transfer sense. We claim that frozen DermLIP + linear scaffold reaches AUROC 0.944 ± 0.003 on `strong_held_out_2` under our specific evaluation protocol, with HAM10000 contamination unpartitioned. The honest reading is that this is consistent with two hypotheses (dermoscopy-domain transfer; HAM10000 overlap) and that EXP-009 is needed to choose between them.

### 8. Required follow-up work (EXP-009) (≈0.5 page)

A dedicated section, not buried in conclusion. Structure:

- **§8.1 The partition question.** State precisely what is unresolved. The DermLIP win is consistent with two distinct hypotheses; the BiomedCLIP partition rules out a third (broader medical pretraining); the dermoscopy-vs-HAM10000-contamination distinction remains.
- **§8.2 EXP-009 design.** Self-pretrain a DINOv2 ViT-B/14 on a non-HAM10000 dermoscopy corpus assembled from ISIC archives minus HAM10000, DermNet, BCN20000 non-HAM10000 components, and similar sources. Run a short JEPA-style or MIM objective (~20 epochs). Run the EXP-004 recipe on top.
- **§8.3 Decision table for EXP-009 outcome.**

  | EXP-009 test AUROC on `strong_held_out_2` | Interpretation | Effect on the paper claim |
  |---|---|---|
  | ≥ 0.85 | Dermoscopy-domain transfer is sufficient; HAM10000 contamination was not the driver of EXP-007. | The DermLIP headline survives, and the contamination caveat downgrades to "we verified this isn't the cause." |
  | 0.50 – 0.80 | Partial transfer; some of EXP-007's win was contamination, some was domain. | The paper claim becomes "domain pretraining helps, but not fully without dataset overlap." |
  | 0.30 – 0.50 | Most of EXP-007's win was HAM10000 contamination. | The paper headline narrows to "JEPA + frozen backbone unlocks when the backbone has seen the eval dataset; cleanly out-of-dataset frozen backbones don't." |

- **§8.4 Why we did not run EXP-009 before submission.** Corpus assembly is multi-week (provenance audit on every ISIC component to confirm HAM10000 exclusion; SSL pretrain compute; pretrain-quality-check on a held-out HAM10000 split). We chose to share the methodology and the nine-run arc at this stage rather than block on EXP-009. The paper is explicit that the partition question is open.
- **§8.5 What feedback we are seeking.** Whether the proxy-task design is reasonable; whether the three-family disjointness is the right granularity; whether the partition we have proposed is the right one or whether a different partition (e.g. a dermoscopy-text-only retrieval probe over DermLIP) would be more informative; what other follow-ups would matter for the next stage.

### 9. Limitations (≈0.5 page)

1. **Pretraining contamination is unpartitioned.** Central caveat. EXP-009 (§8) is the experiment that addresses it.
2. **Synthetic nuisance.** `strong`, `strong_held_out`, `strong_held_out_2` are deterministic operations, not real-world acquisition variability.
3. **Single-architecture comparison.** All four backbones are ViT-B-class. Other architectures (DINOv2-style self-distillation, transformer-only image backbones, SigLIP) might behave differently.
4. **Single dataset.** HAM10000 is one dermoscopy benchmark; ISIC archive components and PAD-UFES-20 are unprobed in this work.
5. **HAM10000 is cross-sectional.** No real same-lesion-over-time pairs; the proxy is synthetic throughout.
6. **One scaffold class on the dermoscopy backbone.** Only linear + Adam-MLP ablations on DINOv2; we did not run MLP-on-DermLIP.

### 10. Conclusion (≈0.25 page)

We have characterised when frozen-backbone JEPA-style probes fail and when they succeed under a leakage-controlled, three-family-disjoint nuisance design on HAM10000. Frozen natural-image and frozen general-medical backbones invert below random across two architectures, three scaffolds, and two optimisers. A frozen dermoscopy-pretrained backbone reaches test AUROC 0.944 ± 0.003 — the only above-random result in nine runs — but under a pretraining corpus that almost certainly contains HAM10000. We release the methodology, the run archive, and a scoped partition experiment as required follow-up. We are sharing this preliminary report to validate the methodology and to seek feedback on the partition design before extending the work; we do not claim to have validated or invalidated the underlying thesis at this stage.

### 11. Contributions (target placement: bullet list at end of §1)

1. **Methodological.** A leakage-controlled HAM10000 longitudinal-proxy task with three disjoint synthetic nuisance families designed to isolate "fits the seen distribution" from "generalises out of distribution." Released as code + configs + reproduction launchers.
2. **Empirical (negative).** Across two architectures × three predictor scaffolds × two optimisers, frozen natural-image backbones produce below-random inverted test AUROC on the held-out third family. The failure is robust along scaffold and optimiser axes.
3. **Empirical (positive but unpartitioned).** Frozen DermLIP reaches test AUROC 0.944 ± 0.003 across 5 seeds, +0.364 above the strongest baseline. The win cannot yet be attributed to dermoscopy-domain transfer rather than HAM10000 image-level overlap; EXP-009 (§8) is the open partition.
4. **Empirical (negative-as-partition).** Frozen BiomedCLIP, the cleanest publicly available "general medical pretraining without HAM10000" backbone, lifts only +0.04 over web CLIP, ruling out "any medical-image pretraining" as sufficient.
5. **Reproducibility.** Every primary-tier run is archived publicly with manifests, embeddings, metrics, model card, and logs at `abdelstark/derma-jepa-runs/<run_id>`. A single launcher script reproduces each run end-to-end on a free A10G via Hugging Face Jobs.
6. **A clearly-scoped open question.** EXP-009's design, decision table, and place in the paper are stated up front rather than treated as future work in passing.

---

## 4. Key figures and tables (master list)

| Asset | Source | Where it appears |
|---|---|---|
| Table 1: Cross-run results | EXP-008 §9.4 + seed-sweep summary §4 | §1 (headline), §6 (full) |
| Figure 1: Test AUROC vs pretraining domain | New panel from cross-run table; DermLIP bar carries contamination annotation | §1 / §6 |
| Figure 2: Train-vs-test AUROC scatter | New plot from per-run `metrics.json` | §6 |
| Figure 3: 3-panel JEPA score histograms | `artifacts/plots/jepa_score_histogram.png` from EXP-006b / EXP-007 / EXP-008 | §6 |
| Table 2: Predictor × optimiser ablation | EXP-004 / EXP-005 / EXP-006a §4.1 each | §6 |
| Table 3: Seed sweep | Seed-sweep summary §3 | §6 |
| Table 4: Representation health | per-run §4.5 | Appendix |
| Pipeline diagram | README Mermaid (Pipeline overview) | §3 or §4 |
| Three-family experimental design diagram | README Mermaid (Experimental design) | §3 |
| Predictor-objective diagram | README Mermaid (Method) | §3.1 |
| EXP-009 decision-table | §8.3 above | §8 |

For paper rendering, the Mermaid sources should be re-rendered as TikZ or vector PDF before submission.

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

## 6. Reference list (target ≈25 entries for a preliminary release)

Already in [`README.md` §References](../../README.md). Add for the paper:

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
- **B. Hyperparameters.** Predictor architecture, optimiser, LR, weight decay, batch size, epochs, identity warm-start; for each variant.
- **C. Nuisance-family operation specifications.** Deterministic operation lists for `strong`, `strong_held_out`, `strong_held_out_2`. Exact augmentation-suite source code reference.
- **D. Bootstrap CI protocol.** Per-group resampling, 1,000 samples, 95 % bands.
- **E. Representation-health checks.** Prediction-norm, dimension-variance, collapse flag definitions and per-run values.
- **F. Compute budget.** Per-run wall-clock breakdown; total GPU-hours across all 9 runs + seed sweep.
- **G. Reproducibility checklist.** Following NeurIPS / MLRC standard items, mapped to repository assets.
- **H. Pretraining contamination analysis.** Detailed argument for why HAM10000 is in Derm1M and the boundaries of what EXP-007 / EXP-008 can and cannot conclude. This is a load-bearing appendix for this paper.
- **I. EXP-009 design (planned).** Corpus assembly plan, SSL pretrain protocol, decision table mapping outcome to claim revision (mirrors §8 above with more depth).

---

## 8. Submission targets

In rough order of fit, prioritising venues that accept honest preliminary reports:

| Venue | Track / format | Fit | Notes |
|---|---|---|---|
| arXiv preprint | non-archival | always first | Recommended ahead of any venue submission to lock priority date and to gather feedback. |
| TMLR | full paper, rolling | high | Empirical / methodology track, no page-limit pressure, accepts honest preliminary work. Probably the best primary venue for this paper. |
| MIDL 2026 | short paper (4 pp + refs) | high | Short-paper format is well-suited to the preliminary-evidence framing. |
| ML4H 2026 (NeurIPS workshop) | full or extended-abstract | high | Health-ML community feedback is exactly what we want at this stage. |
| MICCAI 2026 (workshop, e.g. ISIC, MILLanD) | workshop paper | medium-high | Skin-imaging audience. Full MICCAI track is harder without EXP-009. |
| MIDL 2026 (full) | full paper | medium | Full-paper format probably wants EXP-009 done; defer to v2 if EXP-009 lands. |
| NeurIPS 2026 Datasets & Benchmarks | full paper | medium | Methodological contribution and run archive fit, but reviewers will press on contamination. Better as a v2 venue after EXP-009. |
| Full NeurIPS / ICLR | full paper | low for this paper | Wait until EXP-009 closes the partition. |

Concrete plan: **arXiv preprint now, TMLR or ML4H workshop submission within 4 weeks, then a v2 with EXP-009 results for a MIDL or MICCAI full-paper submission.**

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
| 2026-04-28 | AbdelStark + Claude Code | Initial outline. |
| 2026-04-28 | AbdelStark + Claude Code | Reframed as preliminary release. EXP-009 is no longer in scope for this paper; instead it is a dedicated §8 with a decision table. Title, abstract, conclusion, contributions, and submission targets updated to match. New §0 framing block makes the preliminary status explicit. New §7.4 ("What we do not yet claim") and a feedback-seeking paragraph in §8.5. |
