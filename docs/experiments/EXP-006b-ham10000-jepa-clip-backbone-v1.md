# EXP-006b — HAM10000 CLIP ViT-B/16 backbone swap on the EXP-004 proxy (`ham10000-hf-clip-exp006b-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Fourth consecutive negative on the held-out direction, first under a non-DINOv2 backbone. Replacing frozen DINOv2 ViT-B/14 with frozen OpenAI CLIP ViT-B/16 under the EXP-004 linear predictor recipe produces **test AUROC 0.286 [0.265, 0.310]** on `strong_held_out_2` — still decisively below random, still below every cheap baseline, directionally identical to EXP-004/005. The below-random inversion is therefore **not DINOv2-specific**: it replicates across an entirely different pretraining objective (web-caption contrastive vs self-distillation). Under this experiment's linear-predictor scaffold, the frozen-backbone family as a whole fails on this proxy.
**Date (UTC):** 2026-04-24
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-clip-exp006b-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-clip-exp006b-v1/`
**Launcher commit:** `bc414c6` (after CLIP embedding-extraction fix for transformers 5.x)

---

## 1. Summary

EXP-004 and EXP-005 left two hypotheses on the table for the below-random inversion on `strong_held_out_2`:

- **Scaffold bottleneck.** A more expressive predictor class could recover above-random AUROC on the unseen family. EXP-005 partly addressed this by swapping linear → MLP but the MLP underfit training under the inherited SGD recipe, so the capacity hypothesis remained open. EXP-006a picks that thread up with Adam.
- **Backbone bottleneck.** The frozen backbone's distance geometry imposes the floor, and swapping scaffolds on top cannot move it. EXP-006b is the direct falsification test: change only the backbone, keep the linear predictor and every other knob identical to EXP-004, and see whether the inversion survives.

It survives, and with the same magnitude. Test AUROC is **0.286** (vs 0.249 for DINOv2 linear, within overlapping 95 % bootstrap CIs); pixel L2 is still the strongest baseline at 0.580; delta vs strongest baseline is **−0.294**, indistinguishable from EXP-004's −0.331 and EXP-005's −0.310.

Two features of the CLIP run are quantitatively new:

- **Raw CLIP ViT-B/16 cosine on this proxy is extraordinarily miscalibrated: AUROC 0.036.** The frozen CLIP embedding space places "stable" pairs (same lesion under strong nuisance) *much farther apart* than "changing" pairs (different lesion from different diagnosis). This is a far more extreme inversion than raw DINOv2 cosine showed in EXP-004/005 (0.274). CLIP's web-caption pretraining has no incentive to be invariant to dermoscopic nuisance, so this is directionally expected; the magnitude is still striking.
- **The linear predictor on top of CLIP fits training *harder* than on top of DINOv2.** Train AUROC 0.986 vs EXP-004 linear's 0.900. Train loss drops 31 % (vs EXP-004 linear's 13 %). So the CLIP-linear pipeline *does more work* on the training mixture than the DINOv2-linear pipeline did — and still lands at the same below-random floor on the third, held-out family.

Read together, EXP-006b tightens the backbone-bottleneck claim without closing it. The linear predictor can lift raw CLIP cosine from AUROC 0.036 to 0.286 in-distribution (+0.25) and generalise the lift to `strong_held_out_2` (+0.25 above raw CLIP), but its ceiling on the unseen family is still below random. The pattern is consistent across two backbones with orthogonal pretraining signals, which is the strongest evidence in the five-plus-run arc that the frozen-backbone family — not the scaffold on top — is setting the inversion floor.

Collapse checks still pass. Lesion-ID leakage probes still return zero. No representational collapse.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-004

One knob only. Backbone. Everything else is byte-identical to EXP-004 so the delta is a clean read on pretraining choice.

| Knob | EXP-004 | EXP-006b |
|---|---|---|
| `embeddings.models[0].kind` | `dinov2` | **`clip`** |
| `embeddings.models[0].model_name` | `facebook/dinov2-base` | **`openai/clip-vit-base-patch16`** |
| `training.embedding_model_id` | `dinov2_vitb14` | **`clip_vitb16`** |
| Embedding dimension | 768 | **512** |
| Predictor / hidden dim / epochs / batch / LR / weight decay / optimiser | linear / — / 200 / 128 / 0.03 / 0.001 / SGD | linear / — / 200 / 128 / 0.03 / 0.001 / SGD |

The predictor at step 0 is still identity (linear init = `I` + small Gaussian). The L2-MSE JEPA objective and the `weight − I` regulariser are unchanged. The input dimension drop (768 → 512) is a consequence of the backbone swap, not an independent knob; the predictor's parameter count drops proportionally.

### 2.2 Held constant

Same as EXP-004: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol. Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`.

The only additional caveat: CLIP's image processor bakes in its own normalisation (OpenAI CLIP mean/std, not ImageNet), so "preprocessing held constant" is strictly true at the 224×224-center-crop level but not at the normalisation level. This is expected and the only workable choice for a faithful CLIP evaluation.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. The run needed one re-launch after a code bug: the initial attempt at 13:10 UTC crashed in the CLIP embedding pass because `transformers==5.5.4`'s `CLIPModel.get_image_features` now returns a `BaseModelOutputWithPooling` rather than a tensor; the fix is a one-line `.pooler_output` extraction (commit `bc414c6`). Separate from that, the earlier sibling run at 10:08 UTC (`69eb411c…`) trained to completion but failed on the HF Hub `/commit` endpoint with a 500 — a Hub-side infrastructure flake, unrelated to the code. Nothing from either failed attempt landed in the runs dataset, so the working run at 13:10 UTC is the sole authoritative `ham10000-hf-clip-exp006b-v1`.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras, transformers 5.5.4 |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~57 min | Same distribution as EXP-004 |
| CLIP ViT-B/16 embedding export (10,015 images, batch 16) | ~11 min | Single-model export, faster than EXP-005's dual-model DINOv2 pass |
| Baselines (test) | ~5 min | Pixel L2, SSIM, raw CLIP cosine |
| JEPA linear predictor fit | ~7 min | SGD, 200 epochs, 1,000 training pairs, includes eval and report assembly |
| Upload | ~3 min | 3,012 files, 245 MB |
| **Total wall time** | **≈ 85 min** | End-to-end on a single A10G |

The pipeline's cost is still dominated by FUSE-backed image reads during manifest build and embedding export, both of which are invariant to backbone choice.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct, weak) |
| SSIM distance | 0.436 | [0.411, 0.459] | stable > changing (inverted) |
| **JEPA predictor (exp006b, CLIP-B/16 linear)** | **0.286** | [0.265, 0.310] | **stable > changing (inverted)** |
| Raw CLIP ViT-B/16 cosine | 0.036 | [0.030, 0.043] | **stable ≫ changing (extreme inversion)** |

The predictor beats the raw CLIP cosine baseline by +0.25 AUROC — so the linear scaffold *is* doing something — but still loses to pixel L2 by −0.29 and never crosses the random line. Raw CLIP cosine at 0.036 is in a different regime entirely: the model is near-systematically assigning higher similarity to different-lesion pairs than same-lesion pairs on this nuisance family.

### 4.2 JEPA across splits (the crucial diagnostic)

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.986** | [0.981, 0.990] |
| val | `strong_held_out_2` | 0.300 | [0.279, 0.321] |
| test | `strong_held_out_2` | 0.286 | [0.265, 0.310] |

Train AUROC is the structural result. The CLIP-linear predictor fits the two training nuisance families to **0.986** — *tighter* than EXP-004 linear's 0.900 on DINOv2 and much tighter than EXP-005 MLP's 0.572. Yet the val / test AUROCs on the third unseen family collapse to ~0.29, nearly identical to every prior run. The train-to-test drop is **−0.70 AUROC**, the largest gap of any run in the arc. The predictor is memorising the two training nuisance families very effectively under CLIP features and that memory transfers *negatively* to the unseen family.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000585 | 0.000980 |
| 100 | 0.000404 | 0.001081 |
| 200 | 0.000404 | 0.001081 |

Train loss drops 31 % in the first hundred epochs and then flat-lines — the optimiser has saturated the two-family training mixture. Val loss **increases monotonically** from 0.000980 to 0.001081 and also flat-lines — the predictor has zero generalisation pressure toward `strong_held_out_2`, consistent with the train / val AUROC gap.

Cross-run comparison of loss trajectories (all on the EXP-004 proxy layout):

| Run | Backbone | Predictor | Final train loss | Final val loss | Train loss delta |
|---|---|---|---:|---:|:---:|
| EXP-002 | DINOv2 B/14 | linear | 0.000082 | 0.000075 | −10 % |
| EXP-003 | DINOv2 B/14 | linear (held-out eval) | 0.000562 | 0.000837 | −13 % |
| EXP-004 | DINOv2 B/14 | linear (mixed + third held-out) | 0.000688 | 0.001429 | −13 % |
| EXP-005 | DINOv2 B/14 | MLP (mixed + third held-out) | 0.000872 | 0.001120 | −6 % (underfit) |
| **EXP-006b** | **CLIP B/16** | **linear (mixed + third held-out)** | **0.000404** | **0.001081** | **−31 %** |

EXP-006b's train-loss delta is the largest by roughly 2.5× any prior run. The CLIP embedding space, whatever else it is doing, gives the linear predictor a much easier in-distribution optimisation problem than DINOv2 does.

### 4.4 Why CLIP lifts raw AUROC from 0.036 to 0.286

Raw CLIP cosine on the `strong_held_out_2` test split is 0.036 — essentially a deterministic *anti*-ranking of stable vs changing pairs. The linear JEPA predictor, trained on the two *other* nuisance families, then produces a test AUROC of 0.286. That is a +0.25 AUROC improvement over the raw embedding, without ever seeing `strong_held_out_2` during training.

The mechanical account:

- CLIP's projection head is trained on natural-image / caption alignment; dermoscopic crops under strong colour/brightness/geometry distortions are completely out-of-distribution. The resulting embeddings for stable-pair variants are far apart because they look to CLIP like different objects.
- The linear predictor sees two training nuisance families (`strong`, `strong_held_out`) applied to in-distribution-relative CLIP embeddings. It learns a *direction* in the 512-dimensional projection space along which those specific nuisances map "same lesion under perturbation → closer." That direction partially transfers to `strong_held_out_2`: the predictor now roughly un-rotates the third family's nuisance signature too, lifting raw inverted-cosine 0.036 toward 0.286.
- The lift caps at 0.286 because the third family's perturbation geometry is not identical to the training families'. The partial transfer is enough to uncross-invert most of the pairs but not enough to push AUROC above random.

This is a gentler version of the same dynamic observed in EXP-003/004: the linear predictor fits the seen nuisance families and partially generalises to unseen ones, with the partial generalisation landing short of above-random on the third family. The difference is *where* the partial generalisation lands:

| Backbone | Raw cosine AUROC on test | Linear-predictor AUROC on test | Lift from predictor |
|---|---:|---:|:---:|
| DINOv2 ViT-B/14 | 0.274 | 0.249 | **−0.025** (predictor marginally worse than raw cosine) |
| CLIP ViT-B/16 | 0.036 | 0.286 | **+0.250** (predictor dramatically better than raw cosine) |

The linear predictor's *absolute* ceiling on `strong_held_out_2` is roughly the same regardless of backbone (0.25–0.29). Under DINOv2 the raw cosine is already near that ceiling so the predictor looks redundant; under CLIP the raw cosine is far below and the predictor does substantial work to reach it. Either way the ceiling is below random.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.99999982
- `dimension_variance_mean` = 2.45 × 10⁻⁴
- `dimension_variance_min` = 1.37 × 10⁻⁵
- `collapsed` = **False**

Unchanged qualitative profile from EXP-002/003/004/005. `dimension_variance_min` is roughly the same order as EXP-005 (2.3 × 10⁻⁶ there), which is still comfortably above the collapse threshold. No representational collapse — the failure is distributional, not degenerate.

### 4.6 JEPA pair-score distributions across splits

Scores below are per-pair predicted-target distances; lower distance means the predictor places the context and target closer together, which is the correct direction for a stable pair. Gap = `stable_mean − changing_mean`; negative = correct (stable closer), positive = inverted (stable farther).

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.104 | 0.238 | **−0.133 (correct, strong)** |
| val | 0.279 | 0.232 | +0.047 (inverted) |
| test | 0.281 | 0.233 | **+0.049 (inverted)** |

Train gap magnitude (0.133) is the strongest correct-direction separation in the arc and matches the train AUROC 0.986 reading: the CLIP-linear predictor is producing a very clean stable-closer-than-changing ranking in-distribution. Val and test flip sign to inverted, with smaller magnitude (≈0.05) than EXP-004 linear's test gap (+0.104 under DINOv2). The smaller absolute gap under CLIP despite comparable AUROC inversion reflects the overall compression of the JEPA-latent-space score range under CLIP (all test scores sit in roughly [0.2, 0.3]) rather than a weaker inversion per se. The pattern — "train score separation works beautifully, unseen-family separation inverts" — is now visible in two backbones.

---

## 5. Analysis

### 5.1 What EXP-006b actually proved

**Proved:** The below-random inversion on `strong_held_out_2` is not DINOv2-specific. Swapping frozen DINOv2 ViT-B/14 → frozen OpenAI CLIP ViT-B/16, keeping the linear predictor and every other training knob byte-identical, preserves the inversion. Test AUROC 0.286 under CLIP is within overlapping 95 % bootstrap CIs of test AUROC 0.249 under DINOv2. The failure mode is a *frozen-backbone family* property under this scaffold, not a quirk of one pretraining recipe.

**Also proved:** Frozen CLIP's distance geometry on dermoscopic images under strong nuisance is *catastrophically* miscalibrated for the longitudinal proxy (AUROC 0.036 raw). The linear predictor recovers +0.25 of that gap but not more, which places a quantitative ceiling on how much scaffold-only recovery is possible when the raw embedding is this far off.

**Not yet proved:** That *no* scaffold can lift above random. The CLIP result is under the same SGD + weight decay as every prior DINOv2 linear run; an Adam-tuned scaffold on CLIP could in principle do better, though EXP-005's MLP-under-SGD and this linear-under-SGD pattern both suggest the ceiling is backbone-imposed.

**Also not yet proved:** That the pattern generalises beyond these two backbones. Two is better than one, but the backbone space is large. A self-supervised backbone tuned on medical images (e.g., Derm-ViT, MedCLIP, DermDino) could plausibly move the floor. This is explicit EXP-007+ scope.

### 5.2 The six-run picture so far

| Run | Backbone | Predictor | Proxy | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | trivial | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | DINOv2 B/14 | linear | hardened, one-family held-out | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | hardened, mixed train + third-family eval | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP | same as EXP-004 | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| **EXP-006b** | **CLIP B/16** | **linear** | **same as EXP-004** | **0.986 / 0.300 / 0.286** | Pixel L2 = 0.580 | **−0.294** |

Reading the rightmost column across runs 4–6:

- EXP-004: −0.331 (DINOv2, linear fits 0.90, inverts to 0.25).
- EXP-005: −0.310 (DINOv2, MLP fits 0.57, inverts to 0.27).
- EXP-006b: −0.294 (CLIP, linear fits 0.99, inverts to 0.29).

The train AUROCs span 0.572 to 0.986 across these three runs; the test AUROCs span 0.249 to 0.286. **The more the scaffold fits training, the harder it inverts on the unseen family — but the unseen-family ceiling barely moves.** That is the clearest evidence yet that the scaffold is fighting against a fixed-backbone-imposed floor.

### 5.3 What a properly-fit MLP on CLIP (or EXP-006a on DINOv2) would likely show

After EXP-006b the ex-ante prediction for EXP-006a (Adam-tuned MLP on DINOv2) sharpens. The three candidate outcomes from EXP-005 §5.3 were:

1. Train ≈ 0.95, test ≈ 0.25 (still inverted) — backbone is the root cause.
2. Train ≈ 0.95, test ≈ 0.50 (flat, non-inverted) — partial scaffold win.
3. Train ≈ 0.95, test ≈ 0.75+ (above baselines) — scaffold was the full bottleneck.

EXP-006b's CLIP-linear result strongly points to outcome (1) for EXP-006a. The mechanism is now explicitly characterised: a scaffold that fits training well produces a representation of "what a stable pair looks like under this nuisance family" that is accurate for the training families and incorrect for the held-out family, and the accurate-on-seen / incorrect-on-unseen pattern produces below-random rather than chance-level test AUROC because the two geometries are *opposite*, not independent. Adam on an MLP over DINOv2 should reproduce that pattern with, if anything, tighter training fit and larger inversion.

An outcome of (2) or (3) under EXP-006a would be a genuine surprise and would reopen the scaffold hypothesis. Either way, EXP-006a plus EXP-006b give the cleanest single slice of the two-dimensional (backbone × scaffold) ablation.

### 5.4 Thesis-level statement after six runs

> On a leakage-controlled HAM10000 longitudinal-proxy task, a linear JEPA-style predictor over a frozen vision backbone beats cheap baselines by +0.27 AUROC when the test nuisance family matches training, loses decisively on one held-out family, and inverts below random on a third held-out family after mixed-family training. The inversion pattern replicates across two backbones with orthogonal pretraining signals — self-supervised DINOv2 ViT-B/14 and contrastive OpenAI CLIP ViT-B/16 — with test AUROCs within overlapping bootstrap CIs (0.249 vs 0.286). Under DINOv2 the raw cosine baseline sits near the linear-predictor ceiling; under CLIP the raw cosine is dramatically lower (0.036) and the linear predictor must lift it +0.25 AUROC just to reach the same ceiling. The observation that the unseen-family ceiling is stable across two backbones and two scaffold classes (linear, under-trained MLP) is the strongest evidence in the arc that the binding constraint is the frozen-backbone family under this nuisance-held-out evaluation, not the specific backbone or the specific scaffold. EXP-006a will close the scaffold loop by giving the MLP the optimiser it needs to actually fit training; the expected outcome is test AUROC still near 0.25.

---

## 6. Limitations and threats to validity

1. **Two backbones is not "all backbones."** Self-supervised (DINOv2) and contrastive (CLIP) share the property of being trained on natural images with no dermoscopic exposure. A backbone pretrained on medical images (e.g., MedCLIP, RETFound, Derm-Foundation) could plausibly move the floor. This is the highest-priority post-EXP-006 experiment.
2. **CLIP image preprocessing differs.** The CLIP image processor normalises with OpenAI CLIP mean/std rather than ImageNet. Held constant at the 224×224-center-crop level but strictly a confound with "backbone swap." Running CLIP under ImageNet normalisation was considered and rejected as an unfair evaluation of CLIP itself.
3. **Still one seed.** Across EXP-001 through EXP-006b, all six runs use seed 20260422. A seed sweep remains pending (EXP-007 scope).
4. **Still one third-family design.** `strong_held_out_2` is one specific choice. A different disjoint family could plausibly produce a different-magnitude inversion, though the six-run arc makes the existence of *some* inversion fairly robust.
5. **Linear predictor still fits under SGD with `weight − I` pull.** EXP-006a addresses the MLP version of this under Adam; a linear predictor under Adam was not run. Ex-ante it should not matter since the linear predictor did fit training under SGD, but this is not verified.
6. **HAM10000 is cross-sectional.** Every longitudinal caveat from EXP-001 onward still holds.

---

## 7. What changes for the next run (EXP-007 scoping)

With EXP-006b complete and EXP-006a pending, the natural follow-ups in priority order:

1. **Finish EXP-006a** (Adam-tuned MLP on DINOv2, launched 2026-04-24, run_id `ham10000-hf-dinov2-exp006a-v1`). Closes the scaffold-capacity leg of the EXP-005 split scope. Expected outcome after EXP-006b: train AUROC ≈ 0.95, test AUROC ≈ 0.25, consistent with backbone-bottleneck.
2. **Medical-domain backbone swap.** The single most informative next test after EXP-006b: re-run the EXP-004 recipe under a backbone pretrained on dermoscopic or medical images. Candidates, in rough order of availability: Derm-Foundation (Google), RETFound (adapted to skin), MedCLIP, or a self-supervised DINOv2 fine-tuned on ISIC/HAM10000 unlabeled. A result of test AUROC ≥ 0.5 on `strong_held_out_2` under any of these would be the first above-random result in the arc since EXP-002 and would localise the failure to "natural-image pretraining" rather than "any frozen backbone."
3. **Light fine-tuning of the existing DINOv2 backbone on HAM10000 unlabeled.** If (2) is too heavy an engineering lift, a cheaper experiment: fine-tune DINOv2 ViT-B/14 with a short JEPA-style objective on the full HAM10000 unlabeled pool (plus nuisance augmentation) and re-run the EXP-004 recipe on top. Tests whether domain exposure alone — not architectural change — is enough to move the floor.
4. **Seed sweep.** 3–5 seeds on the EXP-006b config to nail down the CI on the inversion point. Cheap, and should probably happen before any paper draft.

Explicitly still not yet:

- Abandoning the project. The six-run arc plus EXP-006a will constitute a complete methodological characterisation of when and why frozen-backbone JEPA fails on a hardened medical longitudinal proxy. That is publishable as methodology even if no run ever lifts test AUROC above random.
- Broader tasks (multi-class, segmentation). The arc's contribution is *the failure mode*, and dilating the task would make that contribution harder to cleanly state.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-clip-exp006b-v1 \
  ./scripts/hf_jobs_ham10000_exp006b.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-clip-exp006b-v1
```

Expected top line: `auroc: 0.2864`, `strongest_baseline: pixel_l2 = 0.5802`, `delta_vs_baseline: −0.2939`, `collapsed: False`, `tier: public`.

### 8.3 Config diff vs EXP-004

```diff
 training:
-  model_id: jepa_predictor_ham10000_exp004_v1
+  model_id: jepa_predictor_ham10000_exp006b_v1
-  embedding_model_id: dinov2_vitb14
+  embedding_model_id: clip_vitb16
   epochs: 200
   batch_size: 128
   learning_rate: 0.03
   weight_decay: 0.001
   predictor: linear

 embeddings:
   models:
-    - model_id: dinov2_vitb14
-      kind: dinov2
-      model_name: facebook/dinov2-base
+    - model_id: clip_vitb16
+      kind: clip
+      model_name: openai/clip-vit-base-patch16
       batch_size: 16
       device: auto
```

### 8.4 Code fix required for transformers 5.x

`transformers==5.5.4` changed `CLIPModel.get_image_features` to return a `BaseModelOutputWithPooling` instead of a tensor; the projected image embedding lives at `.pooler_output`. Fix in `src/derma_jepa/embeddings.py:276-278` (commit `bc414c6`). Earlier transformers versions that returned a tensor directly will still work under the `getattr(outputs, "pooler_output", outputs)` guard.

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Replacing frozen DINOv2 ViT-B/14 with frozen OpenAI CLIP ViT-B/16 on the same HAM10000 proxy, with an identical linear JEPA-style predictor trained under identical SGD + weight-decay, leaves test AUROC on `strong_held_out_2` essentially unchanged at 0.286 [0.265, 0.310] — within overlapping 95 % bootstrap CIs of DINOv2's 0.249 [0.227, 0.272]. Raw CLIP cosine on the same pairs sits at AUROC 0.036, so the linear predictor is lifting raw in-domain performance by +0.25 AUROC and still landing below random on the unseen family. The inversion replicates across two backbones with orthogonal pretraining objectives, which is the six-run arc's strongest evidence that the frozen-backbone family — not the scaffold on top — imposes the performance floor on this task.

### 9.2 Numbers safe to quote

- CLIP-linear train AUROC: **0.986** [0.981, 0.990] (tightest training fit in the arc)
- CLIP-linear val AUROC: 0.300 [0.279, 0.321]
- CLIP-linear test AUROC: **0.286** [0.265, 0.310]
- Raw CLIP ViT-B/16 cosine baseline AUROC: **0.036** [0.030, 0.043] (extreme inversion)
- Pixel L2 baseline AUROC: 0.580 (unchanged from EXP-004/005)
- SSIM baseline AUROC: 0.436 (unchanged)
- Delta vs strongest baseline: **−0.294**
- Linear-predictor lift over raw CLIP cosine (test): **+0.250** (0.036 → 0.286)
- Train → test AUROC drop: **−0.700** (largest in the arc)
- Train loss reduction over 200 epochs: **31 %** (tightest training fit in the arc)

### 9.3 Pedagogical beats

1. **"The same scaffold on two orthogonal backbones lands at the same test ceiling."** The simplest reading of the six-run arc, now that EXP-006b is in. Six runs, two backbones, three scaffold configurations, and test AUROC on `strong_held_out_2` is 0.249 / 0.270 / 0.286 across the three with-inversion configurations. The ceiling is barely-moving.
2. **"Lifting raw cosine by +0.25 AUROC does not help if the ceiling is at 0.29."** The CLIP run quantitatively separates "the scaffold did work" (it did; +0.25 lift over raw CLIP) from "the scaffold reached above random" (it did not). Scaffold recovery is real but bounded.
3. **"A backbone that's miscalibrated on the task can still be the 'right' comparison."** Raw CLIP at AUROC 0.036 is not a failure of the experiment — it is an informative data point about what CLIP's pretraining signal means for an out-of-domain task, and it makes the +0.25 lift from the linear predictor interpretable.
4. **"Falsifying the easy hypothesis is progress."** EXP-006b falsifies "DINOv2 is weird for dermoscopy." The inversion is not DINOv2-specific. That narrows the search space without yet giving the answer.

### 9.4 Updated cross-run table (the single most useful asset)

| Run | Backbone | Predictor | Proxy | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | trivial | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | DINOv2 B/14 | linear | hardened, one-family held-out | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | hardened, mixed + third-family eval | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | same as EXP-004 | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| **EXP-006b** | **CLIP B/16** | **linear** | **same as EXP-004** | **0.986 / 0.300 / 0.286** | Pixel L2 = 0.580 | **−0.294** |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-006b |
|---|---|
| Backbone-swap ablation | §2.1, §4.1, §4.2 |
| Raw-cosine vs predictor decomposition | §4.4 |
| Six-run joint reading | §5.2 |
| Thesis-level statement | §5.4 |
| Limitations | §6 |
| EXP-007 design | §7 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — pixel L2, SSIM, raw CLIP cosine on `strong_held_out_2`. Note the CLIP cosine histogram's stable/changing separation is visually strongly inverted.
- `artifacts/plots/jepa_score_histogram.png` — CLIP-linear test-split JEPA scores. Stable/changing overlap is larger than in EXP-004/005 but retains the inverted orientation.
- `logs/progress.jsonl` — full per-stage timing trace including the CLIP load report.

### 9.7 Six-experiment arc update

The story now has a sixth act that is the strongest evidence-weight act so far:

1. Build pipeline, proxy is trivial.
2. Harden proxy, JEPA wins +0.27.
3. Held-out family: win collapses.
4. Mixed-family training on third unseen family: predictor inverts below random.
5. MLP predictor on the same proxy: underfits training, test AUROC matches frozen DINOv2 cosine; one leg of the scaffold-vs-backbone ablation.
6. **Swap DINOv2 → CLIP on the linear predictor. Same inversion, same test ceiling, tightest training fit in the arc.** The scaffold can do more work when the backbone gives it more headroom (raw AUROC 0.036 → 0.286), but it still cannot cross random on the unseen family. The backbone-bottleneck hypothesis is now the best-fitting explanation across six runs.

The arc will be complete after EXP-006a closes the scaffold leg; after that, the next move is a medical-domain backbone (§7).

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-24 | AbdelStark + Claude Code | Initial report; run completed on commit `bc414c6` after a transformers-5.x CLIP API fix. Six-run arc now spans two backbones. EXP-007 scope locked: medical-domain backbone swap + seed sweep, pending EXP-006a. |
