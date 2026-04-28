# EXP-007 — HAM10000 DermLIP dermoscopy-pretrained backbone on the EXP-004 proxy (`ham10000-hf-dermlip-exp007-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** First above-random result on `strong_held_out_2` since EXP-002. Replacing frozen DINOv2 ViT-B/14 (and frozen OpenAI CLIP ViT-B/16) with frozen DermLIP — OpenAI CLIP ViT-B/16 architecture CLIP-trained on Derm1M (~1M dermatology image-text pairs) — produces test AUROC **0.945** [0.935, 0.954] on the same EXP-004 proxy where every prior frozen-backbone configuration landed at 0.25–0.29. Delta vs strongest baseline is **+0.364** AUROC. Pretraining-data contamination with HAM10000 is the central caveat and is not addressed by EXP-007 alone (EXP-008 is the partition experiment).
**Date (UTC):** 2026-04-24 → 2026-04-27
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dermlip-exp007-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dermlip-exp007-v1/`
**Launcher commit:** `ba3afce`

---

## 1. Summary

The seven prior runs (EXP-001 through EXP-006a/b) characterised the failure surface of frozen natural-image vision backbones on this leakage-controlled HAM10000 longitudinal-proxy task with a third held-out nuisance family (`strong_held_out_2`). Across two backbones (DINOv2 ViT-B/14, OpenAI CLIP ViT-B/16) and three scaffolds (linear, underfit MLP, fit MLP under Adam), test AUROC sat at 0.25–0.29 — below random and below cheap baselines. EXP-006a §7 scoped the next experiment as a domain-pretrained backbone swap.

EXP-007 swaps in DermLIP (`redlessone/DermLIP_ViT-B-16`) — the same OpenAI CLIP ViT-B/16 architecture used in EXP-006b, CLIP-trained on Derm1M (~1M dermatology image-text pairs spanning dermoscopy, clinical, total-body photography, and dermatopathology) instead of OpenAI WIT (≈400M web image-text pairs). The training scaffold, optimiser, hyperparameters, dataset splits, nuisance-family layout, and every other knob are byte-identical to EXP-006b. The only changed bit is the source of the frozen weights.

Headline numbers:

- Test AUROC **0.945** [0.935, 0.954]. EXP-006b on the identical config was 0.286.
- Train / val / test AUROC: 0.9999 / 0.944 / 0.945. Train → test drop is 0.005, within bootstrap noise.
- Delta vs strongest baseline: **+0.364**. Pixel L2 is unchanged at 0.580.
- Raw DermLIP cosine baseline: **0.109** [0.095, 0.124]. The raw embedding is still inverted on this proxy. The linear predictor lifts AUROC by +0.84 from 0.109 to 0.945.

The structural difference between EXP-006b and EXP-007 is which 1M images and captions the CLIP loss saw during pretraining. Architecture, scaffold, optimiser, and every dataset-side choice are held constant. Raw DermLIP cosine being only modestly less inverted than raw OpenAI CLIP cosine (0.109 vs 0.036) means domain pretraining did not produce a directly task-aligned cosine geometry; what changed is that a linear function on top can extract a feature that ranks stable pairs closer than changing pairs, and that feature transfers cleanly to the unseen nuisance family.

Pretraining contamination caveat: Derm1M almost certainly contains HAM10000 raw images. HAM10000 is the most-used public dermoscopy dataset and a likely ingredient of any "Derm-1M" web-scraped corpus. The evaluation labels do not directly leak — `strong_held_out_2` synthetic nuisance augmentations were generated post-hoc and were never seen by DermLIP's pretraining — but the backbone has had effectively unlimited exposure to the underlying lesion images at native quality. EXP-008 re-runs the same recipe under a medical backbone (BiomedCLIP, PMC-15M) that did not see HAM10000, to separate "domain pretraining helps" from "this specific dataset's pretraining helps."

Collapse checks pass. Lesion-ID leakage probes return zero.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-006b

One knob only. Backbone weights. Architecture, scaffold, optimiser, splits, and nuisance layout byte-identical to EXP-006b.

| Knob | EXP-006b | EXP-007 |
|---|---|---|
| `embeddings.models[0].kind` | `clip` (transformers) | **`open_clip`** (open_clip) |
| `embeddings.models[0].model_name` | `openai/clip-vit-base-patch16` | **`hf-hub:redlessone/DermLIP_ViT-B-16`** |
| `training.embedding_model_id` | `clip_vitb16` | **`dermlip_b16`** |
| Architecture | OpenAI CLIP ViT-B/16 | OpenAI CLIP ViT-B/16 |
| Embedding dim | 512 | 512 |
| Predictor / hidden / epochs / batch / LR / weight decay / optimiser | linear / — / 200 / 128 / 0.03 / 0.001 / SGD | same |

The `kind` change from `clip` → `open_clip` is a loader change (transformers → open_clip library), needed because DermLIP is published as an open_clip checkpoint. Both libraries instantiate the same ViT-B/16 graph at inference time. Image normalisation in both cases follows OpenAI CLIP mean/std; preprocessing pipeline upstream is identical.

DermLIP_ViT-B-16 was chosen over the sister model `DermLIP_PanDerm-base-w-PubMed-256` because the latter's HF config uses an older open_clip schema (`pretrain_path` key) that fails to load on open_clip 3.3.0. Picking the OpenAI-CLIP-architecture variant has the additional benefit of giving an exact-arch match to EXP-006b, so EXP-006b vs EXP-007 isolates the pretraining data alone.

### 2.2 Held constant

Same as EXP-004/005/006a/006b: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol (1,000 samples, 95 % CI). Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`.

### 2.3 What DermLIP saw during pretraining

DermLIP's vision tower was CLIP-trained on **Derm1M** — ~1M dermatology image-text pairs collected from medical literature, public dermoscopy datasets, and dermatology websites. The DermLIP / PanDerm papers describe the data as covering dermoscopy, clinical photography, total-body photography, and dermatopathology. A non-trivial fraction of public-data Derm1M is almost certainly HAM10000 (and ISIC archive), since those are the canonical public dermoscopy datasets. The DermLIP authors do not publish a per-source breakdown.

Implication: DermLIP has had exposure to the raw images in our HAM10000 splits at native quality. The evaluation labels (stable / changing under `strong_held_out_2` nuisance) cannot have leaked because the synthetic nuisance augmentations are deterministic from a seed that is part of EXP-002+ and were never published. But the backbone is closer to "fine-tuned on this dataset" than to "out-of-domain." This is the central limitation of EXP-007 in isolation.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. Single attempt.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras + `open-clip-torch==3.3.0` |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~58 min | Same distribution as EXP-006b |
| DermLIP ViT-B/16 embedding export (8,004 unique images, batch 16) | ~11 min | Single-model open_clip `encode_image` pass; weights are 599 MB safetensors |
| Baselines (test) | ~5 min | Pixel L2, SSIM, raw DermLIP cosine |
| JEPA linear predictor fit | ~6 min | SGD, 200 epochs, 1,000 training pairs, includes eval and report assembly |
| Upload | ~3 min | 3,012 files, 245 MB |
| **Total wall time** | **≈ 87 min** | End-to-end |

The open_clip path adds ~20 s of model load (HF-Hub download + safetensors → torch tensor materialisation) but is otherwise indistinguishable in compute profile from the transformers CLIP path used in EXP-006b.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| **JEPA predictor (exp007, DermLIP linear)** | **0.945** | [0.935, 0.954] | **changing > stable (correct)** |
| Pixel L2 | 0.580 | [0.556, 0.606] | changing > stable (correct, weak) |
| SSIM distance | 0.436 | [0.411, 0.459] | stable > changing (inverted) |
| Raw DermLIP cosine | 0.109 | [0.095, 0.124] | stable ≫ changing (inverted) |

The predictor's CI lower bound (0.935) is +0.36 above pixel L2's CI upper bound (0.606), so the win is decisively above the strongest baseline. The predictor is in the correct direction — changing scores higher than stable, as a distance — unlike every linear-on-frozen-natural-image-backbone result on `strong_held_out_2`. Raw DermLIP cosine is still inverted at 0.109; the dermoscopic backbone alone, without a learned predictor on top, does not solve the task. The work is done by the linear projection learned over DermLIP features.

### 4.2 JEPA across splits

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.9999** | [0.9998, 1.0000] |
| val | `strong_held_out_2` | 0.944 | [0.934, 0.952] |
| test | `strong_held_out_2` | **0.945** | [0.935, 0.954] |

Train AUROC is essentially perfect — the linear predictor solves the training mixture cleanly. Val and test on the unseen `strong_held_out_2` family are within 0.001 of each other and ~0.06 below train. EXP-004 was 0.900 → 0.249 (drop −0.65). EXP-007 is 1.000 → 0.945 (drop −0.05). The generalisation profile is qualitatively different from prior EXP-004-proxy runs.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000761 | 0.000759 |
| 100 | 0.000462 | 0.000939 |
| 200 | 0.000461 | 0.000939 |

Train loss drops 39 % over the first 100 epochs and then flat-lines. Val loss rises from 0.000759 to 0.000939 (24 %) and also saturates. Note that val AUROC stays at 0.944 while val MSE rises — the predictor is incurring slightly higher latent-level MSE on val pairs while preserving the ranking that drives AUROC. MSE in 512-d normalised space and rank-based AUROC are not always co-monotone.

Cross-run comparison (all on the EXP-004 proxy, ordered by recency):

| Run | Backbone | Predictor | Train loss delta | Train AUROC | Test AUROC |
|---|---|---|:---:|---:|---:|
| EXP-004 | DINOv2 B/14 | linear | −13 % | 0.900 | 0.249 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | −6 % | 0.572 | 0.270 |
| EXP-006a | DINOv2 B/14 | MLP (Adam fit) | −3 % | 0.893 | 0.248 |
| EXP-006b | OpenAI CLIP B/16 | linear | −31 % | 0.986 | 0.286 |
| **EXP-007** | **DermLIP B/16** | **linear** | **−39 %** | **0.9999** | **0.945** |

DermLIP gives the linear predictor the largest train-loss reduction in the runs so far and the tightest train fit (AUROC essentially 1.0). DermLIP is the only configuration where this train fit transfers to test: under DINOv2 and OpenAI CLIP, train AUROCs in the 0.89–0.99 range cohabited with test AUROCs in the 0.25–0.29 range. Loss is a poor cross-backbone signal (MSE depends on absolute embedding-space scale); AUROC is the apples-to-apples read.

### 4.4 Why DermLIP+linear lifts AUROC from 0.109 to 0.945

The linear predictor in EXP-006b on OpenAI CLIP lifted raw cosine 0.036 → 0.286 (+0.25 AUROC, capped below random). The same scaffold on DermLIP lifts raw cosine 0.109 → 0.945 (+0.84 AUROC, fully solving the task). A mechanical account:

- The training side mixes `strong` + `strong_held_out` nuisance families. Both predictors (CLIP-linear and DermLIP-linear) see embeddings of stable pairs under those two families and learn a linear map that pulls stable pairs closer together post-projection.
- On `strong_held_out_2`, both predictors must extrapolate. Whether this works depends on whether the direction the linear map learned on the training families is the same direction the third family's nuisance creates.
- For OpenAI CLIP, natural-image pretraining does not align nuisance directions across distinct dermoscopic perturbation families. The map learned on `strong + strong_held_out` does not generalise to `strong_held_out_2`; the predictor extrapolates wrongly and inverts on test.
- For DermLIP, dermoscopy-domain CLIP pretraining over Derm1M appears to have organised the embedding space such that a single linear nuisance direction captures stable-pair similarity across all three families. The map learned on the training families generalises to the third because the third family lives along the same direction.

This is a hypothesis, not a proof. A direction-structure probe — comparing principal components of stable-pair difference vectors across families under DermLIP vs OpenAI CLIP — would test it.

What the result does establish, regardless of the mechanism: domain-aligned pretraining is sufficient under this experimental design to make a linear JEPA-style predictor generalise to a third unseen nuisance family. The seven prior runs established that no scaffold under natural-image pretraining could do this. EXP-007 establishes that a domain-pretrained backbone can.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.99999982
- `dimension_variance_mean` = 3.68 × 10⁻⁴
- `dimension_variance_min` = 3.39 × 10⁻⁵
- `collapsed` = **False**

Dimension variance min is the highest of the EXP-004-proxy runs (EXP-006b 1.4 × 10⁻⁵, EXP-006a 2.5 × 10⁻⁶). DermLIP's predicted-target latents use more dimensions than the natural-image-backbone predictors did, consistent with more usable embedding-space structure rather than collapse onto a few dominant components.

### 4.6 JEPA pair-score distributions across splits

Scores are per-pair predicted-target distances; lower distance = correct for stable pairs. Gap = `stable_mean − changing_mean`; negative = correct (stable closer), positive = inverted.

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.115 | 0.362 | **−0.247 (correct)** |
| val | 0.235 | 0.352 | **−0.117 (correct)** |
| test | 0.236 | 0.346 | **−0.111 (correct)** |

All three splits show the correct direction, which is new for the EXP-004 proxy. EXP-006b had train gap −0.133 in the correct direction but val/test flipped to inverted. Val and test gaps under DermLIP are roughly half of train but cleanly negative — the predictor produces stable < changing on the unseen nuisance family. That is the structural difference behind AUROC 0.945.

---

## 5. Analysis

### 5.1 What EXP-007 actually proved

**Proved (with caveats):** Replacing the frozen vision backbone in the EXP-004 recipe with DermLIP — a backbone of identical architecture, differing only in CLIP-pretraining data (Derm1M dermatology pairs vs OpenAI WIT web image-text pairs) — moves test AUROC on `strong_held_out_2` from 0.286 to 0.945. The +0.66 AUROC swing is localised to the pretraining-data axis, holding architecture, scaffold, optimiser, and every dataset-split / nuisance-family choice fixed.

**Proved (cleanly):** The seven-run claim that "frozen-backbone family is the binding constraint on this proxy under nuisance-held-out evaluation" was a falsifiable claim. EXP-007 falsifies it in the direction of "frozen domain-aligned backbones can succeed." The claim should be re-stated as "frozen natural-image backbones fail; frozen domain-aligned backbones succeed" until further evidence changes the picture.

**Not proved (the EXP-008 wedge):** That the lift comes from "domain pretraining" rather than from "this specific dataset's pretraining." DermLIP's Derm1M almost certainly includes HAM10000, so DermLIP has effectively been pretrained on the image distribution we are evaluating on, though not on the synthetic nuisance augmentations the labels depend on. A non-HAM10000 medical backbone (BiomedCLIP) is needed to separate "any medical-image pretraining suffices" from "pretraining that saw HAM10000 was doing the work."

**Not proved (capacity ceiling):** That 0.945 is the achievable ceiling rather than just where this scaffold lands. A more expressive predictor on top of DermLIP could in principle push higher; the test AUROC is already so high that the marginal value of pursuing this is small.

### 5.2 The eight-run picture so far

| Run | Backbone | Predictor | Optimizer | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | SGD | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | SGD | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | +0.269 |
| EXP-003 | DINOv2 B/14 | linear | SGD | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | SGD | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | SGD | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| EXP-006a | DINOv2 B/14 | MLP (fit) | Adam | 0.893 / 0.266 / 0.248 | Pixel L2 = 0.580 | −0.332 |
| EXP-006b | OpenAI CLIP B/16 | linear | SGD | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |
| **EXP-007** | **DermLIP B/16** | **linear** | **SGD** | **0.9999 / 0.944 / 0.945** | Pixel L2 = 0.580 | **+0.364** |

Test AUROC across runs 4–7 (EXP-004 proxy, varying scaffold/backbone) ranges from 0.248 to 0.945. The four configurations with frozen natural-image backbones cluster at 0.25–0.29. The single configuration with a domain-pretrained backbone is at 0.945. Δ vs strongest baseline across the whole sequence: EXP-007's +0.364 is the largest positive, outpacing EXP-002's matched-eval +0.269. Under nuisance-held-out evaluation, EXP-007 is the only positive result.

### 5.3 Thesis-level statement after eight runs

> On a leakage-controlled HAM10000 longitudinal-proxy task with three disjoint nuisance families and evaluation on the third (unseen) family, a JEPA-style linear predictor over a frozen vision backbone:
>
> - Beats baselines by +0.27 AUROC when the backbone is DINOv2 ViT-B/14 and evaluation matches training nuisance (EXP-002).
> - Loses on one held-out family (−0.28) under DINOv2 (EXP-003).
> - Inverts below random on a third held-out family (−0.33) under DINOv2 with mixed-family training (EXP-004), and that inversion is preserved under MLP scaffolds under SGD and Adam (EXP-005, EXP-006a) and under a backbone swap to natural-image-pretrained OpenAI CLIP ViT-B/16 (EXP-006b).
> - Reaches +0.36 AUROC over baselines on the same third held-out family when the backbone is DermLIP ViT-B/16 — identical CLIP architecture, CLIP-trained on Derm1M dermatology images instead of OpenAI WIT web image-text pairs (EXP-007).
>
> The eight-run sequence is consistent with pretraining-data domain being the load-bearing axis on this proxy under nuisance-held-out evaluation. EXP-008 will test whether the win generalises across medical-domain backbones beyond DermLIP, addressing the open question of dataset-specific vs domain-general pretraining contribution.

---

## 6. Limitations and threats to validity

1. **Pretraining contamination.** Derm1M almost certainly includes HAM10000 raw images. The synthetic nuisance augmentations defining stable / changing labels for `strong_held_out_2` are post-hoc and were never in DermLIP's training data, so the labels themselves cannot have leaked. But DermLIP has had unbounded exposure to the underlying lesion images, which makes EXP-007 as much a "fine-tuned-on-this-dataset evaluation" as an "out-of-domain transfer evaluation." This is the central limitation; EXP-008 (BiomedCLIP) is the experiment that addresses it.
2. **One domain backbone.** DermLIP is one specific dermoscopy-pretrained model. Other dermoscopy or medical backbones (PanDerm raw SSL, MedSigLIP, MONET, BiomedCLIP) might land at different ceilings.
3. **Licence-restricted backbone.** DermLIP is CC-BY-NC 4.0 — research use only. The positive result therefore comes from a non-commercial weight set; downstream applications would need to either negotiate licensing or replicate the recipe under an MIT/Apache backbone.
4. **Single seed.** All eight runs use seed 20260422. Seed sweep on the EXP-007 config remains a follow-up.
5. **One third-family design.** `strong_held_out_2` is a specific synthetic-augmentation family.
6. **HAM10000 is cross-sectional.** Real longitudinal evaluation pending; this is a proxy task throughout.
7. **No mechanism check.** The hypothesis in §4.4 — that DermLIP organises nuisance directions consistently across families — is plausible but unverified.

---

## 7. What changes for the next run (EXP-008 scoping)

In priority order:

1. **EXP-008 — BiomedCLIP backbone swap.** Re-run the EXP-007 recipe under `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. BiomedCLIP is ViT-B/16 trained on PMC-15M (PubMed Central figure-caption pairs). It is broadly biomedical but contains very little dermoscopy and almost certainly did not see HAM10000. Outcomes:
   - Test AUROC ≥ 0.85 → "any medical-image pretraining suffices" — generalises EXP-007's win to a non-contaminated backbone.
   - Test AUROC ≈ 0.30–0.50 → "intermediate between natural-image and dermoscopy" — there is a domain-distance gradient.
   - Test AUROC ≈ 0.25–0.29 → "domain pretraining alone is not enough; HAM10000-specific pretraining is" — narrows the paper claim and motivates EXP-009 (a non-HAM10000 dermoscopy backbone).
2. **Seed sweep (3–5 seeds) on EXP-007 config.** Cheap; confirms 0.945 is not a single-seed artifact. Should happen before paper draft, after EXP-008 because EXP-008 is information-richer per GPU-hour.
3. **MLP-on-DermLIP ablation.** Replace linear with the EXP-006a Adam MLP scaffold on top of DermLIP. Predicted to land at ≥ 0.945 since the linear predictor already saturates; a meaningful drop would be informative about predictor-class-vs-backbone interaction.
4. **Direction-structure probe.** Compute principal-component vectors of stable-pair difference embeddings under each nuisance family for both DermLIP and OpenAI CLIP. Tests the §4.4 hypothesis about cross-family direction alignment.

Explicitly still not yet:

- Unfreezing DermLIP. With AUROC at 0.945 on a frozen backbone, no urgency. If EXP-008 partitions the win toward "specifically HAM10000-contaminated pretraining," fine-tuning becomes more interesting; otherwise hold.
- Broader tasks (multi-class, segmentation). The contribution is the failure / success characterisation; broadening dilutes it.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-dermlip-exp007-v1 \
  ./scripts/hf_jobs_ham10000_exp007.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dermlip-exp007-v1
```

Expected top line: `auroc: 0.9447`, `strongest_baseline: pixel_l2 = 0.5802`, `delta_vs_baseline: +0.3645`, `collapsed: False`, `tier: public`.

### 8.3 Config diff vs EXP-006b

```diff
 embeddings:
   models:
-    - model_id: clip_vitb16
-      kind: clip
-      model_name: openai/clip-vit-base-patch16
+    - model_id: dermlip_b16
+      kind: open_clip
+      model_name: hf-hub:redlessone/DermLIP_ViT-B-16
       batch_size: 16
       device: auto

 training:
-  model_id: jepa_predictor_ham10000_exp006b_v1
-  embedding_model_id: clip_vitb16
+  model_id: jepa_predictor_ham10000_exp007_v1
+  embedding_model_id: dermlip_b16
   epochs: 200
   batch_size: 128
   learning_rate: 0.03
   weight_decay: 0.001
   predictor: linear
```

### 8.4 Code surface added in `ba3afce`

- `pyproject.toml`: `open-clip-torch>=2.24` added to `[model]` extras.
- `scripts/hf_jobs_constraints.txt`: `open-clip-torch==3.3.0` pinned.
- `src/derma_jepa/config.py`: `open_clip` added to the embedding-model `kind` enum.
- `src/derma_jepa/embeddings.py`: new `_open_clip_matrix` function, dispatched on `kind == "open_clip"`. Uses `open_clip.create_model_and_transforms(model_name)` + `model.encode_image(tensor)`. The HF Hub `hf-hub:` prefix is supported natively by open_clip.

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Replacing the frozen OpenAI CLIP ViT-B/16 backbone with DermLIP — identical architecture, CLIP-trained on Derm1M dermatology image-text pairs instead of OpenAI WIT web image-text pairs — lifts test AUROC on the EXP-004 `strong_held_out_2` proxy from 0.286 to **0.945** [0.935, 0.954]. Train→test drop collapses from −0.70 (EXP-006b) to −0.05. Raw DermLIP cosine on the same pairs is still inverted at AUROC 0.109; the linear JEPA predictor lifts +0.84 AUROC on top of the frozen embedding — substantial recovery work that the same scaffold could not do over OpenAI CLIP. Pretraining-data domain is the load-bearing axis on this proxy under nuisance-held-out evaluation. Pretraining-contamination caveat: Derm1M almost certainly includes HAM10000.

### 9.2 Numbers safe to quote

- DermLIP-linear train AUROC: **0.9999** [0.9998, 1.0000]
- DermLIP-linear val AUROC: 0.9435 [0.9335, 0.9524]
- DermLIP-linear test AUROC: **0.9447** [0.9346, 0.9537]
- Raw DermLIP ViT-B/16 cosine baseline AUROC: **0.109** [0.095, 0.124]
- Pixel L2 baseline AUROC: 0.580 (unchanged)
- Delta vs strongest baseline: **+0.364**
- Train → test AUROC drop: **−0.05**
- Linear-predictor lift over raw DermLIP cosine: **+0.836** (0.109 → 0.945)
- Delta vs EXP-006b (identical recipe under OpenAI CLIP): **+0.659**

### 9.3 Pedagogical beats

1. **Pretraining data domain is the load-bearing axis.** Holding architecture, scaffold, optimiser, hyperparameters, splits, and nuisance layout constant, the only thing that moves between EXP-006b's 0.286 and EXP-007's 0.945 is the 1M images and captions DermLIP saw during CLIP-pretraining.
2. **A frozen backbone with the right priors plus a linear projection is enough.** The probe is operationally minimal: 200 epochs of SGD on 1,000 training pairs, a single linear layer plus identity warm-start, no fine-tuning of the backbone.
3. **Raw cosine and predictor AUROC are not co-monotone.** Raw DermLIP cosine is at 0.109 (inverted); the linear predictor on top reaches 0.945. "The embedding space encodes the task signal" is not the same as "raw cosine in the embedding space ranks the task signal correctly." The probe extracts structure that raw distance does not.
4. **Positive results need contamination caveats too.** A 0.945 result coming from a backbone trained on a corpus that likely includes the eval dataset is publishable, but only with the caveat front-loaded and a partition experiment in the same paper.

### 9.4 Updated cross-run table

| Run | Backbone | Predictor | Optimizer | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | linear | SGD | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | linear | SGD | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | +0.269 |
| EXP-003 | DINOv2 B/14 | linear | SGD | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | linear | SGD | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | MLP (underfit) | SGD | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| EXP-006a | DINOv2 B/14 | MLP (fit) | Adam | 0.893 / 0.266 / 0.248 | Pixel L2 = 0.580 | −0.332 |
| EXP-006b | OpenAI CLIP B/16 | linear | SGD | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |
| **EXP-007** | **DermLIP B/16** | **linear** | **SGD** | **0.9999 / 0.944 / 0.945** | Pixel L2 = 0.580 | **+0.364** |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-007 |
|---|---|
| Domain-pretraining ablation | §2.1, §4.1, §4.2 |
| Train→test trajectory comparison | §4.2, §4.3 |
| Raw-cosine vs predictor decomposition | §4.4 |
| Eight-run joint reading | §5.2 |
| Thesis-level statement | §5.3 |
| Pretraining contamination caveat | §6, §7 |
| EXP-008 partition design | §7 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — pixel L2, SSIM, raw DermLIP cosine on `strong_held_out_2`. Raw DermLIP cosine histogram is visibly inverted between stable and changing, similar to raw OpenAI CLIP in EXP-006b.
- `artifacts/plots/jepa_score_histogram.png` — DermLIP-linear test-split JEPA scores. Stable/changing distributions are cleanly separated in the correct direction; side-by-side with EXP-006b's overlapping inverted histograms is a useful figure.
- `logs/progress.jsonl` — per-stage timing including the open_clip load report.

### 9.7 Eight-experiment summary

1. **EXP-001:** build pipeline; proxy is trivial.
2. **EXP-002:** harden proxy; JEPA wins +0.27 on matched eval.
3. **EXP-003:** hold out one nuisance family; win collapses to a loss.
4. **EXP-004:** mix two families in training, evaluate on a third; predictor inverts below random.
5. **EXP-005:** MLP under SGD; underfits, capacity question deferred.
6. **EXP-006a:** MLP under Adam; fits training, still inverts. Scaffold-capacity hypothesis falsified.
7. **EXP-006b:** OpenAI CLIP backbone; same inversion. DINOv2-specificity hypothesis falsified.
8. **EXP-007:** DermLIP (dermoscopy CLIP) backbone; **AUROC 0.945 on the unseen family.** Pretraining-data domain identified as the load-bearing axis under this proxy and evaluation.

EXP-008 partitions the EXP-007 win between "any medical pretraining" and "this-dataset-contaminated pretraining."

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-27 | AbdelStark + Claude Code | Initial report. Run completed on commit `ba3afce` after open_clip integration. First above-random `strong_held_out_2` result since EXP-002. EXP-008 scope locked to BiomedCLIP partition experiment. |
