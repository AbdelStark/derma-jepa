# EXP-006b — HAM10000 CLIP ViT-B/16 backbone swap on the EXP-004 proxy (`ham10000-hf-clip-exp006b-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Fourth consecutive negative on the held-out direction, first under a non-DINOv2 backbone. Replacing frozen DINOv2 ViT-B/14 with frozen OpenAI CLIP ViT-B/16 under the EXP-004 linear-predictor recipe leaves test AUROC on `strong_held_out_2` at **0.286 [0.265, 0.310]** — overlapping 95 % bootstrap CIs with DINOv2 linear's 0.249 [0.227, 0.272]. The below-random inversion is therefore not DINOv2-specific.
**Date (UTC):** 2026-04-24
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-clip-exp006b-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-clip-exp006b-v1/`
**Launcher commit:** `bc414c6`

---

## 1. Summary

EXP-004 and EXP-005 left two open hypotheses for the below-random inversion observed on `strong_held_out_2`:

- **Scaffold bottleneck.** A more expressive predictor could recover above-random AUROC on the unseen family. EXP-005 swapped linear → MLP but the MLP underfit under the inherited SGD recipe; EXP-006a re-tests with Adam.
- **Backbone bottleneck.** The frozen backbone's distance geometry sets the floor, and swapping scaffolds cannot move it. EXP-006b tests this by changing only the backbone, keeping the linear predictor and every other knob identical to EXP-004.

The inversion survives the swap. Test AUROC is **0.286** [0.265, 0.310]. Pixel L2 is still the strongest baseline at 0.580. Delta vs the strongest baseline is **−0.294**, within bootstrap noise of EXP-004's −0.331 and EXP-005's −0.310.

Two new observations under CLIP:

- Raw CLIP ViT-B/16 cosine on this proxy is far below random at AUROC 0.036, much lower than raw DINOv2 cosine (0.274 in EXP-004/005). CLIP's web-caption pretraining gives it no incentive to be invariant to dermoscopic nuisance.
- The linear predictor fits training harder over CLIP than over DINOv2: train AUROC 0.986 vs 0.900, train-loss reduction 31 % vs 13 %. Despite the tighter training fit, test AUROC stays in the same 0.25–0.29 band as every prior linear-on-frozen-natural-image run on this proxy.

The linear predictor lifts raw CLIP cosine from 0.036 to 0.286 — a +0.25 in-distribution lift that partially transfers to `strong_held_out_2` but does not cross random. The same scaffold on two backbones with different pretraining objectives lands at similar test AUROC on the held-out family, which is consistent with the frozen backbone, rather than the predictor, setting the floor.

Collapse checks pass. Lesion-ID leakage probes return zero.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-004

One knob only. Backbone. Everything else byte-identical to EXP-004.

| Knob | EXP-004 | EXP-006b |
|---|---|---|
| `embeddings.models[0].kind` | `dinov2` | **`clip`** |
| `embeddings.models[0].model_name` | `facebook/dinov2-base` | **`openai/clip-vit-base-patch16`** |
| `training.embedding_model_id` | `dinov2_vitb14` | **`clip_vitb16`** |
| Embedding dimension | 768 | 512 |
| Predictor / hidden / epochs / batch / LR / weight decay / optimiser | linear / — / 200 / 128 / 0.03 / 0.001 / SGD | same |

The predictor at step 0 is still identity (linear init = `I` + small Gaussian). The L2-MSE JEPA objective and the `weight − I` regulariser are unchanged. The input dimension drop (768 → 512) is a consequence of the backbone swap, not an independent knob; the predictor's parameter count drops proportionally.

### 2.2 Held constant

Same as EXP-004: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol. Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`.

CLIP's image processor uses OpenAI CLIP mean/std rather than ImageNet, so "preprocessing held constant" is true at the 224×224-center-crop level but not at the normalisation level. Running CLIP under ImageNet normalisation was rejected as an unfair evaluation of CLIP itself.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. The run needed one re-launch after a code bug: the initial attempt at 13:10 UTC crashed in the CLIP embedding pass because `transformers==5.5.4`'s `CLIPModel.get_image_features` returns a `BaseModelOutputWithPooling` rather than a tensor; the fix is a one-line `.pooler_output` extraction (commit `bc414c6`). A separate earlier attempt at 10:08 UTC (`69eb411c…`) trained to completion but failed on the HF Hub `/commit` endpoint with a 500 — a Hub-side infrastructure flake, unrelated to the code. Neither failed attempt landed in the runs dataset, so the 13:10 UTC run is the sole authoritative `ham10000-hf-clip-exp006b-v1`.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras, transformers 5.5.4 |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~57 min | Same distribution as EXP-004 |
| CLIP ViT-B/16 embedding export (10,015 images, batch 16) | ~11 min | Single-model export |
| Baselines (test) | ~5 min | Pixel L2, SSIM, raw CLIP cosine |
| JEPA linear predictor fit | ~7 min | SGD, 200 epochs, 1,000 training pairs, includes eval and report assembly |
| Upload | ~3 min | 3,012 files, 245 MB |
| **Total wall time** | **≈ 85 min** | End-to-end on a single A10G |

Cost is dominated by FUSE-backed image reads during manifest build and embedding export, both invariant to backbone choice.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct, weak) |
| SSIM distance | 0.436 | [0.411, 0.459] | stable > changing (inverted) |
| **JEPA predictor (exp006b, CLIP-B/16 linear)** | **0.286** | [0.265, 0.310] | **stable > changing (inverted)** |
| Raw CLIP ViT-B/16 cosine | 0.036 | [0.030, 0.043] | stable ≫ changing (inverted) |

The predictor beats raw CLIP cosine by +0.25 AUROC, so the linear scaffold extracts some signal, but still loses to pixel L2 by −0.29 and stays below random. Raw CLIP cosine at 0.036 indicates a near-systematic assignment of higher similarity to different-lesion pairs than to same-lesion pairs on this nuisance family.

### 4.2 JEPA across splits

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.986** | [0.981, 0.990] |
| val | `strong_held_out_2` | 0.300 | [0.279, 0.321] |
| test | `strong_held_out_2` | 0.286 | [0.265, 0.310] |

The CLIP-linear predictor fits the two training nuisance families to AUROC 0.986, higher than EXP-004 linear's 0.900 and EXP-005 MLP's 0.572. Val and test AUROC on the third unseen family fall to ~0.29, in the same band as every prior run. Train-to-test drop is −0.70 AUROC. The predictor fits the two training nuisance families well, and that fit transfers in the wrong direction to the unseen family.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000585 | 0.000980 |
| 100 | 0.000404 | 0.001081 |
| 200 | 0.000404 | 0.001081 |

Train loss drops 31 % in the first hundred epochs and then flat-lines. Val loss rises from 0.000980 to 0.001081 and also flat-lines.

Cross-run loss trajectories on the EXP-004 proxy:

| Run | Backbone | Predictor | Final train loss | Final val loss | Train loss delta |
|---|---|---|---:|---:|:---:|
| EXP-002 | DINOv2 B/14 | linear | 0.000082 | 0.000075 | −10 % |
| EXP-003 | DINOv2 B/14 | linear (held-out eval) | 0.000562 | 0.000837 | −13 % |
| EXP-004 | DINOv2 B/14 | linear (mixed + third held-out) | 0.000688 | 0.001429 | −13 % |
| EXP-005 | DINOv2 B/14 | MLP (mixed + third held-out) | 0.000872 | 0.001120 | −6 % (underfit) |
| **EXP-006b** | **CLIP B/16** | **linear (mixed + third held-out)** | **0.000404** | **0.001081** | **−31 %** |

EXP-006b's train-loss reduction is roughly 2.5× any prior run's. The CLIP embedding space gives the linear predictor an easier in-distribution optimisation problem than DINOv2 does.

### 4.4 Why CLIP lifts raw AUROC from 0.036 to 0.286

Raw CLIP cosine on `strong_held_out_2` is 0.036 — essentially a deterministic anti-ranking of stable vs changing pairs. The linear predictor, trained on the two other nuisance families, then produces test AUROC 0.286, a +0.25 AUROC improvement over the raw embedding without ever seeing `strong_held_out_2` during training.

A mechanical account:

- CLIP's projection head was trained on natural-image / caption alignment; dermoscopic crops under strong colour/brightness/geometry distortions are out-of-distribution. The resulting embeddings for stable-pair variants land far apart because they look to CLIP like different objects.
- The linear predictor sees two training nuisance families (`strong`, `strong_held_out`) applied to in-distribution-relative CLIP embeddings. It learns a direction in the 512-d projection space along which those specific nuisances map "same lesion under perturbation → closer." That direction partially transfers to `strong_held_out_2`: the predictor un-rotates the third family's nuisance signature too, lifting raw inverted-cosine 0.036 toward 0.286.
- The lift caps at 0.286 because the third family's perturbation geometry is not identical to the training families'. Partial transfer is enough to uncross-invert most pairs but not enough to push AUROC above random.

This is a milder version of the dynamic observed in EXP-003/004: the linear predictor fits the seen nuisance families and partially generalises to unseen ones, with the partial generalisation landing short of above-random on the third family. The difference from DINOv2 is where the partial generalisation lands relative to the raw cosine baseline:

| Backbone | Raw cosine AUROC on test | Linear-predictor AUROC on test | Lift from predictor |
|---|---:|---:|:---:|
| DINOv2 ViT-B/14 | 0.274 | 0.249 | −0.025 |
| CLIP ViT-B/16 | 0.036 | 0.286 | +0.250 |

The predictor's absolute ceiling on `strong_held_out_2` is roughly the same under both backbones (0.25–0.29). Under DINOv2 the raw cosine is already near that ceiling so the predictor looks redundant; under CLIP the raw cosine is far below and the predictor does substantial work to reach the same point. Either way the ceiling is below random.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.99999982
- `dimension_variance_mean` = 2.45 × 10⁻⁴
- `dimension_variance_min` = 1.37 × 10⁻⁵
- `collapsed` = **False**

Same qualitative profile as EXP-002/003/004/005. `dimension_variance_min` is the same order as EXP-005 (2.3 × 10⁻⁶), comfortably above the collapse threshold. No representational collapse.

### 4.6 JEPA pair-score distributions across splits

Scores are per-pair predicted-target distances; lower distance means the predictor places context and target closer, which is the correct direction for a stable pair. Gap = `stable_mean − changing_mean`; negative = correct (stable closer), positive = inverted.

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.104 | 0.238 | **−0.133 (correct)** |
| val | 0.279 | 0.232 | +0.047 (inverted) |
| test | 0.281 | 0.233 | **+0.049 (inverted)** |

Train gap (0.133, correct direction) matches the train AUROC 0.986. Val and test flip to inverted at +0.05, smaller magnitude than EXP-004 linear's test gap (+0.104 under DINOv2). The smaller absolute gap under CLIP reflects compression of the predicted-target distance range (test scores cluster in [0.2, 0.3]) rather than weaker inversion per se. The pattern — strong correct separation on training families, inversion on the held-out family — replicates across two backbones.

---

## 5. Analysis

### 5.1 What EXP-006b actually proved

**Proved:** The below-random inversion on `strong_held_out_2` is not DINOv2-specific. Swapping frozen DINOv2 ViT-B/14 → frozen OpenAI CLIP ViT-B/16, keeping the linear predictor and every other training knob byte-identical, preserves the inversion. Test AUROC 0.286 under CLIP is within overlapping 95 % bootstrap CIs of test AUROC 0.249 under DINOv2.

**Also proved:** Frozen CLIP's distance geometry on dermoscopic images under strong nuisance is far below random for the longitudinal proxy (raw cosine AUROC 0.036). The linear predictor recovers +0.25 of that gap and no more, which places a quantitative ceiling on scaffold-only recovery when the raw embedding starts that far off.

**Not yet proved:** That no scaffold can lift above random. The CLIP result is under SGD + weight decay, the same recipe as every prior DINOv2 linear run; an Adam-tuned scaffold on CLIP could behave differently, though EXP-005's MLP-under-SGD and this linear-under-SGD pattern both point to the ceiling being backbone-imposed.

**Also not yet proved:** That the pattern generalises beyond these two backbones. A backbone pretrained on medical or dermoscopic images could move the floor; that is explicit EXP-007+ scope.

### 5.2 The six-run picture so far

| Run | Backbone | Predictor | Proxy | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | trivial | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | DINOv2 B/14 | linear | hardened, one-family held-out | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | hardened, mixed train + third-family eval | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP | same as EXP-004 | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| **EXP-006b** | **CLIP B/16** | **linear** | **same as EXP-004** | **0.986 / 0.300 / 0.286** | Pixel L2 = 0.580 | **−0.294** |

Across runs 4–6 (the EXP-004 proxy with varying scaffold/backbone), train AUROC spans 0.572 to 0.986 while test AUROC spans 0.249 to 0.286. The unseen-family test ceiling barely moves regardless of how well the scaffold fits training. The scaffold appears to be fighting a backbone-imposed floor.

### 5.3 What a properly-fit MLP on CLIP (or EXP-006a on DINOv2) would likely show

EXP-005 §5.3 listed three candidate outcomes for a better-tuned MLP:

1. Train ≈ 0.95, test ≈ 0.25 (still inverted) — backbone is the root cause.
2. Train ≈ 0.95, test ≈ 0.50 (flat, non-inverted) — partial scaffold win.
3. Train ≈ 0.95, test ≈ 0.75+ (above baselines) — scaffold was the full bottleneck.

EXP-006b's CLIP-linear result points to outcome (1) for EXP-006a. A scaffold that fits training well produces a representation of "what a stable pair looks like under this nuisance family" that is accurate for the training families and incorrect for the held-out family; that pattern produces below-random rather than chance-level test AUROC because the two geometries are opposite, not independent. Adam on an MLP over DINOv2 should reproduce this with tighter training fit and similar inversion magnitude.

An outcome of (2) or (3) under EXP-006a would reopen the scaffold hypothesis. Either way, EXP-006a + EXP-006b cover one slice of the (backbone × scaffold) ablation.

### 5.4 Thesis-level statement after six runs

> On a leakage-controlled HAM10000 longitudinal-proxy task, a linear JEPA-style predictor over a frozen vision backbone beats cheap baselines by +0.27 AUROC when the test nuisance family matches training, loses on one held-out family, and inverts below random on a third held-out family after mixed-family training. The inversion replicates across two backbones with different pretraining objectives — self-supervised DINOv2 ViT-B/14 and contrastive OpenAI CLIP ViT-B/16 — with overlapping 95 % bootstrap CIs (0.249 vs 0.286). Under DINOv2 the raw cosine baseline sits near the linear-predictor ceiling; under CLIP the raw cosine is much lower (0.036) and the linear predictor lifts it +0.25 AUROC to reach the same ceiling. The unseen-family ceiling is stable across two backbones and two scaffold classes (linear, under-trained MLP), consistent with the frozen-backbone family setting the floor under this nuisance-held-out evaluation. EXP-006a will close the scaffold leg by giving the MLP an optimiser that fits training; the expected outcome is test AUROC near 0.25.

---

## 6. Limitations and threats to validity

1. **Two backbones is not "all backbones."** Self-supervised (DINOv2) and contrastive (CLIP) share the property of being pretrained on natural images with no dermoscopic exposure. A backbone pretrained on medical images could plausibly move the floor.
2. **CLIP image preprocessing differs.** The CLIP image processor normalises with OpenAI CLIP mean/std rather than ImageNet. Held constant at the 224×224-center-crop level but strictly a confound with "backbone swap."
3. **Single seed.** All six runs use seed 20260422.
4. **One third-family design.** `strong_held_out_2` is one specific choice of disjoint family.
5. **Linear predictor under SGD only.** Linear-under-Adam was not run; EXP-006a addresses the MLP-under-Adam case.
6. **HAM10000 is cross-sectional.** Every longitudinal caveat from EXP-001 onward holds.

---

## 7. What changes for the next run (EXP-007 scoping)

In priority order:

1. **Finish EXP-006a** (Adam-tuned MLP on DINOv2). Closes the scaffold-capacity leg of EXP-005 §7. Expected outcome after EXP-006b: train AUROC ≈ 0.95, test AUROC ≈ 0.25.
2. **Medical-domain backbone swap.** Re-run the EXP-004 recipe under a backbone pretrained on dermoscopic or medical images. A test AUROC ≥ 0.5 on `strong_held_out_2` would be the first above-random result since EXP-002 and would localise the failure to "natural-image pretraining" rather than "any frozen backbone."
3. **Domain-adaptation fine-tune of DINOv2.** A cheaper variant of (2): fine-tune DINOv2 ViT-B/14 with a short JEPA-style or MIM objective on HAM10000-unlabeled (plus nuisance augmentation) and re-run the EXP-004 recipe on top.
4. **Seed sweep.** 3–5 seeds on the EXP-006b config. Cheap, useful before any paper draft.

Explicitly still not yet:

- Abandoning the project. The six-run sequence plus EXP-006a will be a complete characterisation of when frozen-backbone JEPA fails on this proxy.
- Broader tasks (multi-class, segmentation). The contribution is the failure mode; broadening dilutes it.

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

`transformers==5.5.4` changed `CLIPModel.get_image_features` to return a `BaseModelOutputWithPooling` instead of a tensor; the projected image embedding lives at `.pooler_output`. Fix in `src/derma_jepa/embeddings.py:276-278` (commit `bc414c6`). Earlier transformers versions that returned a tensor directly still work under the `getattr(outputs, "pooler_output", outputs)` guard.

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Replacing frozen DINOv2 ViT-B/14 with frozen OpenAI CLIP ViT-B/16 on the same HAM10000 proxy, with an identical linear JEPA-style predictor and identical optimiser, leaves test AUROC on `strong_held_out_2` essentially unchanged at 0.286 [0.265, 0.310] — within overlapping 95 % bootstrap CIs of DINOv2's 0.249 [0.227, 0.272]. Raw CLIP cosine on the same pairs sits at 0.036; the linear predictor lifts that by +0.25 AUROC but still lands below random on the unseen family. The inversion replicates across two backbones with different pretraining objectives, consistent with the frozen backbone — not the predictor — setting the floor.

### 9.2 Numbers safe to quote

- CLIP-linear train AUROC: **0.986** [0.981, 0.990]
- CLIP-linear val AUROC: 0.300 [0.279, 0.321]
- CLIP-linear test AUROC: **0.286** [0.265, 0.310]
- Raw CLIP ViT-B/16 cosine baseline AUROC: **0.036** [0.030, 0.043]
- Pixel L2 baseline AUROC: 0.580 (unchanged from EXP-004/005)
- SSIM baseline AUROC: 0.436 (unchanged)
- Delta vs strongest baseline: **−0.294**
- Linear-predictor lift over raw CLIP cosine (test): **+0.250** (0.036 → 0.286)
- Train → test AUROC drop: **−0.700**
- Train loss reduction over 200 epochs: **31 %**

### 9.3 Pedagogical beats

1. **Same scaffold over two backbones with different pretraining objectives lands at the same test ceiling.** Six runs, two backbones, three scaffold configurations: test AUROC on `strong_held_out_2` is 0.249 / 0.270 / 0.286 across the three configurations that train on mixed nuisance.
2. **Lifting raw cosine by +0.25 AUROC does not help if the ceiling is at 0.29.** The CLIP run separates "the scaffold did work" (it lifted raw CLIP +0.25) from "the scaffold reached above random" (it did not).
3. **A backbone that is miscalibrated on the task can still be the right comparison.** Raw CLIP at AUROC 0.036 is an informative data point about what CLIP's pretraining signal means for an out-of-domain task, and it makes the +0.25 lift from the linear predictor interpretable.
4. **Falsifying the easy hypothesis is progress.** The inversion is not DINOv2-specific.

### 9.4 Updated cross-run table

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

- `artifacts/plots/baseline_score_histogram.png` — pixel L2, SSIM, raw CLIP cosine on `strong_held_out_2`. Raw CLIP cosine histogram is visibly inverted between stable and changing.
- `artifacts/plots/jepa_score_histogram.png` — CLIP-linear test-split JEPA scores. Stable/changing distributions overlap more than in EXP-004/005 and remain inverted.
- `logs/progress.jsonl` — per-stage timing trace including the CLIP load report.

### 9.7 Six-experiment summary

1. Build pipeline; proxy is trivial.
2. Harden proxy; JEPA wins +0.27 on matched eval.
3. Hold out one nuisance family; win collapses.
4. Mix two families in training, evaluate on a third; predictor inverts below random.
5. Swap linear → MLP under SGD; MLP underfits, test AUROC ≈ DINOv2 cosine baseline.
6. Swap DINOv2 → CLIP under linear predictor. Same inversion, similar test ceiling, tightest training fit so far. Raw CLIP cosine 0.036 → predictor 0.286; below random on the unseen family.

The next step after EXP-006a is a medical-domain backbone (§7).

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-24 | AbdelStark + Claude Code | Initial report; run completed on commit `bc414c6` after a transformers-5.x CLIP API fix. |
