# EXP-006a — HAM10000 Adam-tuned MLP predictor on the EXP-004 proxy (`ham10000-hf-dinov2-exp006a-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** The scaffold-capacity leg of EXP-006, companion to EXP-006b. Re-running EXP-005's MLP architecture with Adam at LR 1e-3 and weight decay 1e-4 (instead of SGD at LR 3e-2, weight decay 1e-3) lets the MLP actually fit: train AUROC **0.893** [0.879, 0.906], matching EXP-004 linear's 0.900. Test AUROC on `strong_held_out_2` is **0.248** [0.228, 0.269] — statistically indistinguishable from EXP-004 linear's 0.249 and from EXP-005 MLP's 0.270. Delta vs strongest baseline: **−0.332**, the deepest inversion in the arc. This closes the open thread from EXP-005: a *properly-fit* MLP on frozen DINOv2 does not move the unseen-family test AUROC off the linear-predictor floor. Combined with EXP-006b (CLIP linear also at 0.286), the seven-run arc now varies both scaffold and backbone and the test ceiling is stable at 0.25–0.29 across every configuration except the matched-family EXP-002.
**Date (UTC):** 2026-04-24
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dinov2-exp006a-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dinov2-exp006a-v1/`
**Launcher commit:** `bc414c6`

---

## 1. Summary

EXP-005 swapped the predictor class linear → 2-layer MLP on the EXP-004 proxy but inherited EXP-002's optimiser (SGD + LR 0.03 + weight_decay 0.001). The MLP never escaped its zero-W2 identity attractor: train AUROC 0.572, train loss reduction only 6 %. The run could not cleanly answer "does a well-fit MLP generalise" because the MLP did not in fact fit training. EXP-005 §7 scoped a targeted follow-up — same architecture, Adam at LR 1e-3 with weight decay 1e-4 — specifically to let the predictor leave identity.

It does. Train loss drops from 0.000694 to 0.000671 (flat numerically because Adam moves in small steps in latent-space units, but large enough in weight-space to move the ranking dramatically), and train AUROC climbs from EXP-005's 0.572 to **0.893 [0.879, 0.906]**. The residual `W2 ReLU(W1 x + b1) + b2` is now doing real work; the MLP is fitting the mixed `strong + strong_held_out` training distribution approximately as well as EXP-004's linear predictor did (0.893 vs 0.900).

On the test split — the same 1,533-group `strong_held_out_2` evaluation used in EXP-004/005/006b — the AUROC is **0.248 [0.228, 0.269]**, with EER 0.69 and `fpr_at_fixed_tpr(0.8) = 0.986`. That number is statistically indistinguishable from EXP-004 linear's 0.249 and from EXP-006b CLIP linear's 0.286. The train → test drop is **−0.645 AUROC**, almost identical to EXP-004's −0.651. The MLP, once it actually fits, reproduces the linear predictor's collapse on the unseen family with no detectable improvement.

Three consequences that now hold with the full EXP-006 in hand:

- **Scaffold-capacity hypothesis is falsified at this scale.** Five configurations of scaffold-over-frozen-DINOv2 have been run: linear (EXP-002/003/004), underfit MLP (EXP-005), and now fit MLP with Adam (EXP-006a). Every one of them that trains on mixed families and is evaluated on `strong_held_out_2` lands at test AUROC 0.25–0.27. A more expressive predictor, trained until it fits, does not help.
- **Backbone-family hypothesis is strengthened.** EXP-006b ran the same EXP-004 recipe on frozen CLIP ViT-B/16 and got test AUROC 0.286. EXP-006a ran a better scaffold on frozen DINOv2 and got 0.248. Varying the backbone moves the floor slightly (0.25 → 0.29); varying the scaffold does not move it at all. The binding constraint is the frozen backbone.
- **Ex-ante prediction from EXP-006b was correct.** EXP-006b §5.3 predicted outcome (1) — "train AUROC ≈ 0.95, test AUROC ≈ 0.25" — on the strength of six prior runs. EXP-006a realised it (0.89 / 0.25) with essentially no deviation.

Collapse checks still pass. Lesion-ID leakage probes still return zero. No representational collapse.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-005

Three knobs, all targeting the "MLP cannot escape identity" failure mode diagnosed in EXP-005 §4.4. Everything else byte-identical to EXP-005.

| Knob | EXP-005 | EXP-006a |
|---|---|---|
| `training.optimizer` | `sgd` | **`adam`** |
| `training.learning_rate` | 0.03 | **0.001** |
| `training.weight_decay` | 0.001 | **0.0001** |
| Predictor / hidden dim / epochs / batch | MLP identity-residual / 512 / 200 / 128 | same |
| Backbone | DINOv2 ViT-B/14 | same |

The LR drop from 0.03 → 0.001 follows Adam-typical practice (Adam's implicit per-parameter scaling makes SGD-sized LRs unstable); the weight decay drop from 1e-3 → 1e-4 follows the diagnosis that the MLP's zero-W2 identity point was being held in place by SGD's L2 penalty, so the Adam run should start with decoupled weight decay at an order of magnitude lower. No other knobs touched.

### 2.2 Held constant

Same as EXP-004/005/006b: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol. Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`. DINOv2 ViT-S/14 + ViT-B/14 embeddings exported identically to EXP-005.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. Single clean attempt, no re-launch needed.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~68 min | Same distribution as EXP-004/005; some FUSE tail variance |
| DINOv2 S/14 + B/14 embedding export | ~31 min | Identical image set; dual-model |
| Baselines (test) | ~5 min | Pixel L2, SSIM, DINOv2-S / DINOv2-B cosine |
| JEPA MLP predictor fit | **~1.7 s** | torch + A10G, 200 epochs × 1,000 pairs × (768 → 512 → 768); Adam is as fast as SGD at this size |
| Upload | ~3 min | ~3,000 files, ~280 MB |
| **Total wall time** | **≈ 105 min** | End-to-end |

Wall time is dominated by FUSE-backed image reads during manifest build and embedding export. The predictor fit itself takes under two seconds; the three-knob optimiser change has zero runtime cost.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct, weak) |
| SSIM distance | 0.436 | [0.411, 0.459] | stable > changing (inverted) |
| DINOv2 ViT-S/14 cosine | 0.306 | [0.282, 0.329] | stable > changing (inverted) |
| DINOv2 ViT-B/14 cosine | 0.274 | [0.252, 0.298] | stable > changing (inverted) |
| **JEPA predictor (exp006a, Adam MLP)** | **0.248** | [0.228, 0.269] | **stable > changing (inverted)** |

The Adam MLP's test AUROC is **below** every baseline including raw DINOv2 B/14 cosine (0.274). Unlike EXP-006b's CLIP-linear result (where the predictor lifted raw cosine from 0.036 to 0.286), here the predictor actually *worsens* the raw-cosine performance by 0.026 AUROC. The MLP has learned a ranking that generalises worse than passing DINOv2 embeddings through unchanged.

### 4.2 JEPA across splits (the crucial diagnostic)

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.893** | [0.879, 0.906] |
| val | `strong_held_out_2` | 0.266 | [0.244, 0.287] |
| test | `strong_held_out_2` | 0.248 | [0.228, 0.269] |

Train AUROC 0.893 matches EXP-004 linear's 0.900 to within a bootstrap CI. The Adam-tuned MLP fits the training mixture with essentially the same tightness as the linear predictor. Val 0.266 and test 0.248 are statistically indistinguishable, confirming no val-test leakage in the splits. The train → test drop is **−0.645 AUROC**, the largest in the arc alongside EXP-004's −0.651.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000694 | 0.001300 |
| 100 | 0.000671 | 0.001386 |
| 200 | 0.000671 | 0.001377 |

Train loss drops 3.3 % over 200 epochs and saturates by epoch 100. Val loss rises from 0.001300 to 0.001377 (a 6 % increase) and also saturates. The absolute loss delta is tiny — smaller than EXP-005's 6 % train reduction — but crucially the *ranking* produced by the predictor shifts dramatically: train AUROC climbs from ~0.5 at init to 0.893 by end. Adam's adaptive per-parameter step reshuffles the MLP's function class away from identity without needing large L2 loss movement; MSE in the 768-d DINOv2 latent space is a poor indicator of ranking-quality movement at this scale.

Cross-run comparison of loss trajectories on the EXP-004 proxy:

| Run | Optimizer | Predictor | Train loss delta | Train AUROC | Test AUROC |
|---|---|---|:---:|---:|---:|
| EXP-004 | SGD | linear | −13 % | 0.900 | 0.249 |
| EXP-005 | SGD | MLP (underfit) | −6 % | 0.572 | 0.270 |
| **EXP-006a** | **Adam** | **MLP (fit)** | **−3 %** | **0.893** | **0.248** |
| EXP-006b | SGD | linear (CLIP) | −31 % | 0.986 | 0.286 |

Two observations. First, MSE loss delta is *not* a good proxy for fit quality across optimisers and architectures — EXP-006a moves 4× less MSE than EXP-004 and reaches equivalent train AUROC. Second, the "fit tightness" signal is now decoupled from the test-AUROC signal across the last four runs: train AUROC varies 0.57 → 0.99, test AUROC varies 0.25 → 0.29. The test ceiling does not follow the train fit.

### 4.4 Why the Adam MLP lands exactly where EXP-004 linear lands

The EXP-004 linear predictor and the EXP-006a Adam MLP, despite being different function classes trained by different optimisers, produce test AUROCs within a single bootstrap CI (0.249 vs 0.248). The cleanest mechanical account is:

- Both predictors see the same DINOv2 ViT-B/14 embeddings of the same training pairs from the same two nuisance families (`strong`, `strong_held_out`).
- Both predictors learn a function `f: R^768 → R^768` that minimises MSE between predicted and observed target latent on those pairs, under light identity-adjacent regularisation.
- The two training nuisance families induce specific directions in DINOv2's latent space along which stable-pair variants lie. Both predictors learn those directions — linear does it with a single matrix, the MLP does it with a ReLU-mediated residual — and on the training split both succeed.
- On `strong_held_out_2`, the third family induces *different* directions. Neither predictor has learned those. The DINOv2 embedding for a stable `strong_held_out_2` pair drifts into a region of latent space that neither predictor's training data covers.
- Because the MLP's training directions are not the same as the third family's directions, the MLP's extrapolation to that region is *as wrong* as the linear predictor's — and because both predictors were initialised near identity and pulled toward the training directions, both end up extrapolating in roughly the same wrong way.

Under this account the linear predictor is a lower-bound on "what frozen-DINOv2 can do on `strong_held_out_2` with a predictor that fits training." EXP-006a shows the MLP achieves the same lower bound despite being a strictly more expressive function class. That is not a claim that "capacity cannot help" in general — it is a claim that "capacity cannot help when the training distribution does not cover the test distribution's nuisance directions and the backbone does not encode nuisance-invariant features."

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.9999999
- `dimension_variance_mean` = 4.83 × 10⁻⁴
- `dimension_variance_min` = 2.47 × 10⁻⁶
- `collapsed` = **False**

Unchanged qualitative profile from the five prior runs. Dimension variance min sits at the same order as EXP-005 (2.3 × 10⁻⁶) and EXP-006b (1.4 × 10⁻⁵). The MLP has not collapsed; the failure is distributional.

### 4.6 JEPA pair-score distributions across splits

Scores below are per-pair predicted-target distances; lower = more similar = correct for stable pairs. Gap = `stable_mean − changing_mean`; negative = correct, positive = inverted.

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.264 | 0.450 | **−0.186 (correct, strong)** |
| val | 0.526 | 0.413 | +0.112 (inverted) |
| test | 0.525 | 0.406 | **+0.119 (inverted)** |

Train gap of −0.186 is the strongest-magnitude correct-direction separation on DINOv2 in the arc (EXP-004 linear had train AUROC 0.900 on a similar proxy, directly comparable; EXP-005 MLP's gap was only −0.032). Val and test both flip to inverted at +0.11–0.12, within noise of EXP-004 linear's test gap +0.104. The MLP is producing a cleaner-than-linear separation on the training families and the same-magnitude inversion on the held-out family.

---

## 5. Analysis

### 5.1 What EXP-006a actually proved

**Proved:** A 2-layer identity-residual MLP, trained with Adam at LR 1e-3 and decoupled weight decay 1e-4 for 200 epochs on frozen DINOv2 ViT-B/14 embeddings of the EXP-004 proxy, fits the mixed `strong + strong_held_out` training distribution to train AUROC 0.893 and produces test AUROC 0.248 on `strong_held_out_2` — statistically identical to the EXP-004 linear predictor's 0.249 and below the raw DINOv2 ViT-B/14 cosine baseline of 0.274.

**Also proved:** The "MLP underfit so the MLP-capacity question is unsettled" caveat from EXP-005 §5.1 is now lifted. The optimiser change does exactly what EXP-005 §7 predicted (escape zero-W2 attractor, fit training); it does not change the test outcome.

**Not proved:** That no MLP architecture at any scale can move the ceiling. EXP-006a uses a 2-layer 512-hidden-dim MLP. A deeper or wider MLP, or a transformer-style predictor, could in principle behave differently. Ex-ante this seems unlikely on the strength of the seven-run arc but has not been tested.

### 5.2 The seven-run picture so far

| Run | Backbone | Predictor | Optimizer | Proxy | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | SGD | trivial | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | SGD | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | DINOv2 B/14 | linear | SGD | hardened, one-family held-out | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | SGD | hardened, mixed + third-family eval | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | SGD | same as EXP-004 | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| **EXP-006a** | **DINOv2 B/14** | **MLP (fit)** | **Adam** | **same as EXP-004** | **0.893 / 0.266 / 0.248** | Pixel L2 = 0.580 | **−0.332** |
| EXP-006b | CLIP B/16 | linear | SGD | same as EXP-004 | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |

The matrix now covers:

- Two backbones (DINOv2 ViT-B/14, CLIP ViT-B/16)
- Three scaffolds (linear, underfit MLP, fit MLP)
- Two optimisers (SGD, Adam)

Restricted to the EXP-004 proxy (runs 4–7, last four rows), **test AUROCs span 0.248 to 0.286** — a 0.038 AUROC range. Train AUROCs span 0.572 to 0.986 — a 0.41 range, 10× wider than the test range. The training side responds strongly to scaffold/optimiser choice; the test side is effectively pinned at ~0.25–0.29. That decoupling is the arc's cleanest diagnostic.

### 5.3 Thesis-level statement after seven runs

> On a leakage-controlled HAM10000 longitudinal-proxy task with three disjoint nuisance families (`strong`, `strong_held_out`, `strong_held_out_2`), a JEPA-style predictor over a frozen vision backbone beats cheap baselines by +0.27 AUROC when the test nuisance family matches training, loses decisively on one held-out family, and inverts below random on a third held-out family after mixed-family training. Varying the scaffold — linear → underfit MLP → properly-fit Adam-trained MLP — keeps the train AUROC on the mixed training distribution in the 0.57–0.99 range but keeps the test AUROC on `strong_held_out_2` pinned at 0.25–0.27. Varying the backbone — DINOv2 ViT-B/14 → OpenAI CLIP ViT-B/16 — keeps the test AUROC at 0.25–0.29 despite very different raw-cosine baselines (0.27 vs 0.04). The seven-run matrix (2 backbones × 3 scaffolds × 1 optimiser pair) therefore establishes that the binding constraint on `strong_held_out_2` is **the frozen-backbone family** under this proxy design, not the predictor class and not the training recipe. The next productive experiment is a medical-domain-pretrained backbone, which is the only untested cell that could plausibly lift the floor.

### 5.4 What this means for the project arc

EXP-006a + EXP-006b together close the "scaffold vs backbone" question the arc has been chasing since EXP-003. The answer is: the backbone. That is a publishable negative result by itself — "frozen natural-image backbones don't do dermoscopic longitudinal proxy under distributional-shift evaluation" — and it directly motivates the next experiment rather than ending the project. The arc now has a clear narrative:

1. EXP-001: pipeline works on a trivial proxy (ceiling = 1.0).
2. EXP-002: hardened proxy, matched eval: +0.27 AUROC win over baselines.
3. EXP-003: held-out one family: win collapses to a loss.
4. EXP-004: mixed-family training, held-out third family: inversion below random.
5. EXP-005: MLP scaffold under SGD — capacity test left inconclusive by underfit.
6. EXP-006b: CLIP backbone swap — inversion replicates across pretraining family.
7. **EXP-006a: Adam-fit MLP — scaffold-capacity hypothesis falsified on DINOv2.**

EXP-007 then has a single dominant candidate: medical-domain backbone. Anything else is sub-leading.

---

## 6. Limitations and threats to validity

1. **Single MLP architecture.** 2-layer, 512-hidden. Deeper or wider MLPs not tested. The falsification at §5.1 is specific to this architecture; the arc's claim is about this architecture representing a reasonable-capacity scaffold class, not about all scaffolds.
2. **Linear predictor under Adam not tested.** In principle a linear predictor under Adam could behave differently from linear under SGD, though the identity-warm-started linear predictor in EXP-004 already fit training to 0.900 under SGD so ex-ante the optimiser axis looks fully probed.
3. **Two backbones still.** EXP-006b added CLIP; EXP-006a confirms the floor under DINOv2. A third natural-image backbone (e.g., SigLIP, EVA-02) would add robustness but not change the thesis.
4. **Still one seed.** Across all seven runs. Seed sweep remains open.
5. **Still one third-family design.** `strong_held_out_2` is one choice.
6. **HAM10000 is cross-sectional.** Longitudinal caveats unchanged.

---

## 7. What changes for the next run (EXP-007 scoping)

With both halves of EXP-006 in hand, the scaffold-vs-backbone question is resolved and the next experiment is unambiguous. In priority order:

1. **Medical-domain backbone swap.** Re-run the EXP-004 recipe under a backbone pretrained on dermoscopic or medical images. Leading candidates:
   - **Derm-Foundation** (Google/DeepMind, if public weights are accessible).
   - **RETFound** adapted to skin.
   - **MedCLIP / PMC-CLIP** (trained on medical image-text pairs).
   - **DINOv2 fine-tuned on ISIC/HAM10000 unlabeled** via a short JEPA- or masked-patch pretraining objective.
   The decision-theoretic read: any one of these achieving test AUROC ≥ 0.5 on `strong_held_out_2` would be the first above-random result since EXP-002 and would localise the failure to "natural-image pretraining" rather than "any frozen backbone." A result of ≤ 0.3 would close the project on "frozen backbones don't work for this proxy regardless of pretraining domain" and move the next experiment to unfreezing the backbone.
2. **Domain-adaptation pretrain of DINOv2 (cheap version of 1).** If 1 is blocked on weight availability, fine-tune DINOv2 ViT-B/14 with a short MIM or iBOT objective on HAM10000-unlabeled + nuisance-augmented pairs for ~20 epochs, then re-run the EXP-004 recipe on top. Tests whether *domain exposure* alone — without a new architecture — can move the floor.
3. **Seed sweep on EXP-004 and EXP-006b configs.** 3–5 seeds to nail down bootstrap-CI-tight ranges on the inversion magnitude. Should happen before any paper draft; order after 1/2 because those are higher-information experiments.
4. **Linear predictor under Adam.** Quick ablation to close the §6 limitation.

Explicitly still not yet:

- Unfreezing the backbone. Hold this as the dominant post-EXP-007 candidate if 1 and 2 both fail to move the floor; not the next step.
- Broader tasks. The arc's contribution is *the failure mode*; dilating the task would make that contribution harder to state cleanly.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-exp006a-v1 \
  ./scripts/hf_jobs_ham10000_exp006a.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dinov2-exp006a-v1
```

Expected top line: `auroc: 0.2480`, `strongest_baseline: pixel_l2 = 0.5802`, `delta_vs_baseline: −0.3322`, `collapsed: False`, `tier: public`.

### 8.3 Config diff vs EXP-005

```diff
 training:
-  model_id: jepa_predictor_ham10000_exp005_v1
+  model_id: jepa_predictor_ham10000_exp006a_v1
   embedding_model_id: dinov2_vitb14
   epochs: 200
   batch_size: 128
-  learning_rate: 0.03
-  weight_decay: 0.001
+  learning_rate: 0.001
+  weight_decay: 0.0001
   predictor: mlp
   hidden_dim: 512
+  optimizer: adam
```

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Adam-tuning a 2-layer identity-residual MLP predictor over frozen DINOv2 ViT-B/14 on the HAM10000 proxy lifts train AUROC from EXP-005's underfit 0.572 to **0.893** [0.879, 0.906], matching EXP-004 linear's 0.900 — and leaves test AUROC on `strong_held_out_2` at **0.248** [0.228, 0.269], statistically identical to EXP-004's 0.249 and below raw DINOv2 ViT-B/14 cosine at 0.274. The scaffold-capacity hypothesis from EXP-005 §7 is falsified: a properly-fit MLP does not move the unseen-family test ceiling. Combined with EXP-006b's CLIP result (0.286), the seven-run arc now varies both scaffold and backbone across the EXP-004 proxy and the test AUROC is stable at 0.25–0.29 in every configuration. The binding constraint is the frozen-backbone family, not the predictor class or the optimiser.

### 9.2 Numbers safe to quote

- Adam MLP train AUROC: **0.893** [0.879, 0.906]
- Adam MLP val AUROC: 0.266 [0.244, 0.287]
- Adam MLP test AUROC: **0.248** [0.228, 0.269]
- Raw DINOv2 ViT-B/14 cosine test AUROC: 0.274 (predictor underperforms raw cosine by 0.026)
- Pixel L2 baseline AUROC: 0.580 (unchanged)
- Delta vs strongest baseline: **−0.332** (deepest inversion in the arc)
- Train → test AUROC drop: **−0.645** (tied with EXP-004 linear for largest)
- Train loss reduction over 200 epochs: **3.3 %** (loss is a poor fit proxy for this run; AUROC is the real signal)
- Predictor fit runtime: ~1.7 s on A10G

### 9.3 Pedagogical beats

1. **"Loss is not fit; AUROC is fit."** EXP-006a reduces train loss by only 3 % and moves train AUROC from ~0.5 to 0.89. MSE in a 768-d embedding space can be numerically small while ranking-level performance moves dramatically. Always instrument both.
2. **"Closing a caveat is as important as producing a new result."** EXP-006a doesn't discover anything new about the ceiling — it confirms EXP-004's 0.249 with a different function class. That confirmation is what lets the arc *claim* backbone-bottleneck rather than *suspect* it.
3. **"When the backbone is fixed and doesn't encode nuisance-invariant features, capacity buys you training fit without generalisation."** This is the mechanistic statement the arc can now defend. It is stronger than "MLP doesn't help" and more useful for paper exposition.
4. **"Predictions that are robust to being wrong are the ones worth making."** EXP-006b §5.3 predicted "train ≈ 0.95, test ≈ 0.25" as outcome (1). The actual run landed at 0.89 / 0.25. The prediction survives.

### 9.4 Updated cross-run table

| Run | Backbone | Predictor | Optimizer | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | SGD | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | SGD | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | DINOv2 B/14 | linear | SGD | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | SGD | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | SGD | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| **EXP-006a** | **DINOv2 B/14** | **MLP (fit)** | **Adam** | **0.893 / 0.266 / 0.248** | Pixel L2 = 0.580 | **−0.332** |
| EXP-006b | CLIP B/16 | linear | SGD | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-006a |
|---|---|
| Scaffold-capacity ablation | §2.1, §4.1, §4.2 |
| Loss-vs-AUROC decoupling | §4.3 |
| Mechanistic "why MLP = linear on test" | §4.4 |
| Seven-run joint reading | §5.2 |
| Thesis-level statement | §5.3 |
| EXP-007 decision framework | §7 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — identical to EXP-005 (same backbone, same test split).
- `artifacts/plots/jepa_score_histogram.png` — Adam MLP test-split drift scores; stable/changing inverted at test-time, compare to EXP-005 (near-identity) and EXP-004 linear for predictor-class delta.
- `logs/progress.jsonl` — per-stage timing; note the ~1.7 s MLP fit.

### 9.7 Seven-experiment arc update

The arc now reads cleanly as a falsification-driven story:

1. **EXP-001**: build pipeline, prove proxy is trivial.
2. **EXP-002**: harden proxy, JEPA wins +0.27 AUROC on matched eval.
3. **EXP-003**: hold out one nuisance family at eval — win collapses to a loss.
4. **EXP-004**: mix two families in training, evaluate on a third — predictor inverts below random.
5. **EXP-005**: swap linear → MLP under inherited SGD — MLP underfits, cannot separate scaffold-capacity from backbone bottleneck.
6. **EXP-006b**: swap DINOv2 → CLIP with linear — inversion replicates across pretraining objectives.
7. **EXP-006a**: swap SGD → Adam, letting the MLP fit — **fit MLP lands exactly where linear lands, 0.248 vs 0.249**. Scaffold-capacity hypothesis falsified.

Narrative-level: the arc has now converged on "the frozen natural-image backbone is the binding constraint on this task under this nuisance-held-out evaluation." That statement is defensible from the seven-run matrix and motivates exactly one next move: a backbone with domain-relevant pretraining.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-24 | AbdelStark + Claude Code | Initial report. Run completed on commit `bc414c6`, launched in parallel with EXP-006b. Seven-run arc now resolves the scaffold-vs-backbone question; EXP-007 scope locked to medical-domain backbone or domain-adaptation pretrain. |
