# EXP-008 — HAM10000 BiomedCLIP backbone partition for EXP-007 contamination caveat (`ham10000-hf-biomedclip-exp008-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Partial generalisation of EXP-007's win. Test AUROC on `strong_held_out_2` under frozen BiomedCLIP (PMC-15M, broadly biomedical, did not see HAM10000) is **0.325** [0.303, 0.349] — between OpenAI CLIP's 0.286 (EXP-006b, web-pretrained, did not see HAM10000) and DermLIP's 0.945 (EXP-007, Derm1M, almost certainly contains HAM10000), but **much closer to CLIP than to DermLIP**. The +0.04 AUROC lift from "web pretraining" to "general biomedical pretraining" is small; the +0.62 AUROC jump from "general biomedical" to "dermoscopy-specific" carries almost the entire EXP-007 win. EXP-008 therefore **partially weakens the EXP-007 paper headline**: the win is not "domain-aligned medical pretraining unlocks JEPA" — it is "dermoscopy-specific pretraining unlocks JEPA," and EXP-008 cannot separate that from "pretraining that saw HAM10000."
**Date (UTC):** 2026-04-27
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-biomedclip-exp008-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-biomedclip-exp008-v1/`
**Launcher commit:** `c741632`

---

## 1. Summary

EXP-007 reported test AUROC 0.945 on `strong_held_out_2` after swapping the frozen backbone from OpenAI CLIP ViT-B/16 to DermLIP_ViT-B-16 (same architecture, CLIP-trained on Derm1M dermatology image-text pairs). Two competing readings of that result remained open:

- **Domain-aligned-medical-pretraining hypothesis:** any vision encoder pretrained on medical / clinical images would unlock the proxy. DermLIP's win is representative of a broader class.
- **HAM10000-contamination hypothesis:** Derm1M almost certainly contains HAM10000, so DermLIP has effectively been pretrained on the *image distribution* the proxy evaluates on. The win is dataset-specific rather than domain-general.

EXP-008 runs the same EXP-004 recipe under `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` — OpenAI CLIP ViT-B/16 architecture, MIT-licensed, CLIP-trained on PMC-15M (~15M PubMed Central figure-caption pairs spanning radiology, pathology, microscopy, anatomy, and a long tail of clinical figures). PMC-15M is broadly biomedical but contains very little dermoscopy and **does not include HAM10000 / ISIC archive**. It is the cleanest publicly available "domain-aligned medical pretraining without HAM10000 contamination" backbone.

The result lands in the middle band of the EXP-007 §7 decision table:

- **Test AUROC: 0.325** [0.303, 0.349]. Above OpenAI CLIP (0.286) by +0.04 — visible but small. Below DermLIP (0.945) by −0.62 — large.
- **Delta vs strongest baseline: −0.256.** Still inverted, still below pixel L2.
- **Train AUROC: 0.923** [0.911, 0.934]. Tighter training fit than DINOv2 linear (EXP-004) but looser than OpenAI CLIP linear (EXP-006b's 0.99) and DermLIP linear (EXP-007's 0.9999).
- **Train→test drop: −0.60 AUROC.** Nearly as large as EXP-004's −0.65. Generalisation to `strong_held_out_2` does not happen under BiomedCLIP.
- **Raw BiomedCLIP cosine baseline: 0.047** [0.040, 0.055]. Strongly inverted, similar magnitude to OpenAI CLIP's 0.036 and unlike DermLIP's 0.109.

Two-way reading of the three-backbone comparison:

| Backbone | Pretraining | HAM10000 in pretrain? | Test AUROC | Δ vs preceding |
|---|---|---|---:|---:|
| OpenAI CLIP ViT-B/16 (EXP-006b) | LAION web captions | No | 0.286 | — |
| **BiomedCLIP ViT-B/16 (EXP-008)** | **PMC-15M biomedical figures** | **No** | **0.325** | **+0.04** |
| DermLIP ViT-B/16 (EXP-007) | Derm1M dermatology | Likely yes | 0.945 | +0.62 |

The gradient is strongly non-linear. The web → general-medical step adds 0.04 AUROC. The general-medical → dermoscopy step adds 0.62 AUROC. **15× more lift from the dermoscopy-specific step than from the medical-domain step.** That distribution localises the EXP-007 win sharply to "dermoscopy-specific pretraining" rather than to "medical pretraining at all," and EXP-008 alone cannot further partition "dermoscopy-specific transfer" from "HAM10000 image-level overlap."

Collapse checks pass. Lesion-ID leakage probes still return zero. No representational collapse.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-007

One knob only. Pretraining-data axis. Architecture, scaffold, optimiser, splits, nuisance layout, preprocessing, and every other knob byte-identical to EXP-007 (which is itself byte-identical to EXP-006b modulo backbone).

| Knob | EXP-007 | EXP-008 |
|---|---|---|
| `embeddings.models[0].model_name` | `hf-hub:redlessone/DermLIP_ViT-B-16` | **`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`** |
| `training.embedding_model_id` | `dermlip_b16` | **`biomedclip_b16`** |
| Architecture | OpenAI CLIP ViT-B/16 | OpenAI CLIP ViT-B/16 (same) |
| Embedding dim | 512 | 512 (same) |
| `kind` | `open_clip` | `open_clip` (same; BiomedCLIP is a `CustomTextCLIP` under the same loader path) |
| Predictor / hidden / epochs / batch / LR / weight decay / optimiser | linear / — / 200 / 128 / 0.03 / 0.001 / SGD | same |

Both DermLIP and BiomedCLIP are loaded via `open_clip.create_model_and_transforms('hf-hub:...')` and produce a 512-d projected image embedding via `encode_image`. Image normalisation in both cases follows OpenAI CLIP mean/std (the loader's default for ViT-B/16 checkpoints). Preprocessing pipeline upstream is the EXP-004 224×224 center-crop profile.

### 2.2 What BiomedCLIP saw during pretraining

PMC-15M is a corpus of ~15M figure-caption pairs scraped from PubMed Central open-access biomedical articles. Per the BiomedCLIP paper, it spans:

- Radiology (CT, MRI, ultrasound, X-ray): substantial fraction.
- Pathology (microscopy, histology, IHC): substantial fraction.
- Surgical, anatomical, and procedural figures: smaller fraction.
- Dermatology (clinical / dermoscopy): small fraction. The corpus contains figures *referencing* HAM10000 / ISIC datasets in research papers, but does not contain the raw datasets themselves.
- Patient-figure mixtures, charts, diagrams: long tail.

The key qualitative property for EXP-008's purpose: **PMC-15M was not constructed by ingesting HAM10000 or the ISIC archive**. Any HAM10000 image content reaching BiomedCLIP would have to have been downsampled, cropped, or otherwise processed for a published figure, and would represent a tiny fraction (likely < 0.01 %) of the pretraining data. For practical purposes, BiomedCLIP did not see HAM10000.

This is the cleanest publicly-available "biomedical-domain pretraining without HAM10000 contamination" baseline.

### 2.3 Held constant

Same as EXP-004/005/006a/006b/007: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol (1,000 samples, 95 % CI). Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. Single clean attempt.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras + `open-clip-torch==3.3.0` |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~56 min | Same distribution as EXP-007 |
| BiomedCLIP ViT-B/16 embedding export (8,004 unique images, batch 16) | ~10 min | Single-model `encode_image` pass; 784 MB `.bin` checkpoint |
| Baselines (test) | ~5 min | Pixel L2, SSIM, raw BiomedCLIP cosine |
| JEPA linear predictor fit | ~4 min | SGD, 200 epochs, 1,000 training pairs, includes eval and report assembly |
| Upload | ~3 min | 3,012 files, 245 MB |
| **Total wall time** | **≈ 83 min** | End-to-end |

Compute profile is indistinguishable from EXP-007.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct, weak) |
| SSIM distance | 0.436 | [0.411, 0.459] | stable > changing (inverted) |
| **JEPA predictor (exp008, BiomedCLIP linear)** | **0.325** | [0.303, 0.349] | **stable > changing (inverted)** |
| Raw BiomedCLIP cosine | 0.047 | [0.040, 0.055] | stable ≫ changing (extreme inversion) |

The predictor lifts raw BiomedCLIP cosine from 0.047 → 0.325 (+0.28 AUROC) — substantial recovery work, comparable in magnitude to the recovery EXP-006b's predictor did over OpenAI CLIP (0.036 → 0.286, +0.25). Yet the ceiling is still below random. BiomedCLIP looks structurally much more like OpenAI CLIP than like DermLIP from this proxy's perspective, despite being trained on a corpus that is *broadly medical*.

### 4.2 JEPA across splits

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.923** | [0.911, 0.934] |
| val | `strong_held_out_2` | 0.351 | [0.327, 0.375] |
| test | `strong_held_out_2` | **0.325** | [0.303, 0.349] |

Train-to-test drop is −0.60 AUROC. Compare to EXP-007's −0.05 (clean generalisation) and EXP-006b's −0.70 (catastrophic generalisation failure). BiomedCLIP is closer to the failure mode than to the success mode — its in-distribution fit is solid (train 0.92) but does not transfer to the unseen nuisance family.

### 4.3 The three-way pretraining-data comparison (the central diagnostic)

This is the table EXP-008 was designed to produce.

| Run | Backbone | Pretraining corpus | Train AUROC | Val AUROC | Test AUROC | Δ vs strongest |
|---|---|---|---:|---:|---:|---:|
| EXP-006b | OpenAI CLIP B/16 | LAION-400M (web captions) | 0.986 | 0.300 | 0.286 | −0.294 |
| **EXP-008** | **BiomedCLIP B/16** | **PMC-15M (biomedical figures)** | **0.923** | **0.351** | **0.325** | **−0.256** |
| EXP-007 | DermLIP B/16 | Derm1M (dermatology) | 0.9999 | 0.944 | 0.945 | +0.364 |

Three reads:

1. **The gradient is monotone but strongly non-linear.** OpenAI CLIP < BiomedCLIP < DermLIP on test AUROC, in the order of "domain-distance to dermoscopy." But the gap from CLIP to BiomedCLIP is +0.04, and the gap from BiomedCLIP to DermLIP is +0.62. The dermoscopy-specific step carries the bulk of the win.
2. **Train AUROCs do not predict test AUROCs across this row.** OpenAI CLIP fits train hardest (0.986) and lands at 0.286. DermLIP fits train at 0.9999 and lands at 0.945. BiomedCLIP fits train *less hard* than either (0.92) and lands in between. The relationship between train fit and test fit is mediated by the embedding space's structural invariance to nuisance, not by raw representational capacity.
3. **The mechanistic claim from EXP-007 §4.4 is partially confirmed and partially complicated.** EXP-007 hypothesised that DermLIP organises nuisance directions consistently across families. EXP-008 shows BiomedCLIP does *not* organise them this way, despite being a medical-domain encoder. So domain-aligned pretraining alone is insufficient; what matters is whether the corpus contains *enough on-task structure* to align nuisance directions across families. For dermoscopy, that threshold is somewhere between PMC-15M (insufficient) and Derm1M (sufficient).

### 4.4 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000769 | 0.000946 |
| 100 | 0.000557 | 0.001041 |
| 200 | 0.000557 | 0.001041 |

Train loss drops 28 % over the first 100 epochs and flat-lines. Val loss rises 10 % and flat-lines. The shape of the curve is nearly identical to EXP-006b (CLIP linear): the predictor fits the training distribution efficiently, val loss diverges, and val AUROC stays well below random. Same story as the natural-image-backbone runs, with marginally more lift on the test side.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.99999988
- `dimension_variance_mean` = 2.65 × 10⁻⁴
- `dimension_variance_min` = 5.83 × 10⁻⁵
- `collapsed` = **False**

`dimension_variance_min` is the highest of any frozen-natural-image-backbone run on the EXP-004 proxy (EXP-006b was 1.4 × 10⁻⁵; EXP-006a was 2.5 × 10⁻⁶) but well below EXP-007's DermLIP run on the same metric. Consistent with "BiomedCLIP's embedding space has slightly more usable structure than OpenAI CLIP's, but much less than DermLIP's."

### 4.6 JEPA pair-score distributions across splits

Scores are per-pair predicted-target distances; lower distance = more similar = correct for stable pairs. Gap = `stable_mean − changing_mean`; negative = correct (stable closer), positive = inverted.

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.151 | 0.259 | **−0.108 (correct)** |
| val | 0.285 | 0.259 | +0.026 (inverted, weak) |
| test | 0.286 | 0.253 | **+0.033 (inverted, weak)** |

Train gap is the right sign and meaningful magnitude. Val and test gaps flip to inverted but with very small magnitude (~0.03) compared to OpenAI CLIP's (+0.05 test) and DINOv2's (+0.10–0.12). BiomedCLIP is *almost* generalising — its inverted gap on test is the smallest of any frozen-natural-or-general-medical-backbone run — but not quite enough to push test AUROC above 0.5.

---

## 5. Analysis

### 5.1 What EXP-008 actually proved

**Proved:** A frozen vision backbone with general biomedical pretraining (BiomedCLIP, PMC-15M) produces test AUROC 0.325 on the EXP-004 `strong_held_out_2` proxy — slightly better than web-pretrained OpenAI CLIP (0.286) and dramatically worse than dermoscopy-pretrained DermLIP (0.945). The lift from web to general biomedical is small (+0.04); the lift from general biomedical to dermoscopy-specific is large (+0.62).

**Proved:** EXP-007's "domain-aligned pretraining unlocks JEPA" claim does not generalise to "any medical-image pretraining." The win is sharply concentrated at the dermoscopy-specific step.

**Not proved:** Whether the dermoscopy-specific step's win is caused by (a) HAM10000 being part of Derm1M during DermLIP pretraining, or (b) the broader dermoscopy-domain visual structure in Derm1M being sufficient even without HAM10000. EXP-008 closes the "general medical" alternative but cannot disambiguate (a) from (b). A non-HAM10000 dermoscopy-pretrained backbone would be needed; this is harder to come by because every public dermoscopy SSL pretrain known (PanDerm, MONET, DermLIP, …) uses ISIC archives that include HAM10000 as a component.

### 5.2 Updated thesis-level statement after nine runs

The thesis statement from EXP-007 §5.3 now needs the EXP-008 partition built in. Updated form:

> On a leakage-controlled HAM10000 longitudinal-proxy task with three disjoint nuisance families and evaluation on the third (unseen) family, a JEPA-style linear predictor over a frozen vision backbone:
>
> - Wins decisively on matched-eval (DINOv2 ViT-B/14 linear, +0.27 AUROC, EXP-002).
> - Loses on one held-out family (−0.28, EXP-003).
> - Inverts below random on a third unseen family (−0.33, EXP-004), and that inversion is preserved under MLP scaffolds (EXP-005, EXP-006a) and a non-medical backbone swap (OpenAI CLIP, EXP-006b, −0.29).
> - Wins decisively (+0.36) when the backbone is DermLIP, dermoscopy-CLIP-trained on Derm1M which almost certainly includes HAM10000 (EXP-007).
> - **Lifts only marginally (+0.04 over OpenAI CLIP) when the backbone is BiomedCLIP, broadly biomedical-CLIP-trained on PMC-15M which does not include HAM10000 (EXP-008).** The general-medical pretraining is insufficient.
>
> The nine-run sequence therefore characterises a complete success/failure surface for frozen-backbone JEPA-style scaffolds and localises the success regime to a narrow band: **dermoscopy-specific pretraining is required, and a non-trivial fraction of that requirement may be HAM10000 image-level overlap rather than dermoscopy-domain transfer in general.** The honest paper claim is "frozen DermLIP + linear JEPA predictor solves this proxy; general-medical or natural-image frozen backbones do not; the specificity of 'dermoscopy-pretrained' versus 'HAM10000-pretrained' is not yet partitioned."

### 5.3 The three-way comparison and what it means for paper structure

The paper now has a three-act story instead of two:

1. **Failure surface** (EXP-001 → EXP-006a/b): characterise where frozen natural-image JEPA fails. Eight runs, two backbones, three scaffolds.
2. **Apparent fix** (EXP-007): a dermoscopy-pretrained backbone produces a 0.945 test AUROC, dramatic positive result.
3. **Partition** (EXP-008): the apparent fix does not generalise to general-medical pretraining; the win is concentrated at the dermoscopy step. The paper must either narrow its claim (honest) or run an additional experiment to disambiguate dermoscopy-domain transfer from HAM10000 contamination.

This is a *better* paper structure than two acts because it's more honest, and it gives a clean motivating argument for any follow-up experiment that disambiguates further. But it does mean the headline number to lead with is no longer "0.945 from domain-aligned pretraining" — it is "0.945 from dermoscopy pretraining (caveat: contamination with eval images is plausible), with broader medical pretraining insufficient at +0.04 over web."

### 5.4 What EXP-008 tells us about the EXP-007 mechanism

EXP-007 §4.4 hypothesised that DermLIP's embedding space has consistent nuisance directions across all three nuisance families, which is what allows a linear predictor trained on the first two families to generalise to the third. EXP-008 sharpens that hypothesis by ruling out a specific alternative:

- **Ruled out:** "Any embedding space trained on a corpus that contains medical images" has consistent nuisance directions. BiomedCLIP saw 15× more images than DermLIP did, including a non-trivial fraction of dermatology figures, and yet does not exhibit cross-family direction consistency.
- **Still open:** Whether the consistency in DermLIP comes from (a) the dermoscopy-specific visual statistics in Derm1M's non-HAM10000 portion or (b) the HAM10000-specific image content. If a non-HAM10000 dermoscopy SSL run could be assembled, comparing it to DermLIP on this proxy would partition (a) from (b). The next experiment EXP-009 should target this if the paper claim depends on the partition.

---

## 6. Limitations and threats to validity

1. **EXP-008 cannot disambiguate dermoscopy-domain transfer from HAM10000 contamination.** This is the central limitation. The cleanest follow-up requires a non-HAM10000 dermoscopy pretrain, which is harder to obtain than a generic medical or web backbone.
2. **One general-medical backbone.** BiomedCLIP is one specific point in the medical-pretraining design space. MedSigLIP-448 (Google), MONET (Univ. of Washington / Allen AI), or BioMed-CLIP variants might land at different AUROCs. None are likely to reach DermLIP's 0.945 since none have dermoscopy as a primary pretraining target, but they could move the BiomedCLIP point in the partition.
3. **Architecture match still ViT-B/16 OpenAI-CLIP-style.** The three-way table holds architecture constant, which is the right choice for an ablation, but means the comparison cannot speak to whether other architectures (DINOv2-style self-distillation, transformer-only image backbones) would behave differently.
4. **Single seed.** Across all nine runs.
5. **One third-family design.** `strong_held_out_2` is one specific synthetic-augmentation family.
6. **HAM10000 is cross-sectional.** Longitudinal caveats unchanged.

---

## 7. What changes for the next run (EXP-009 scoping)

EXP-008's middle-band result reframes the priority list. The decision-relevant next move is partitioning "dermoscopy-domain transfer" from "HAM10000 contamination." In priority order:

1. **EXP-009 — Self-pretrain DINOv2 on a non-HAM10000 dermoscopy corpus.** Cleanest partition test. Build a corpus of dermoscopic / clinical skin images that explicitly excludes HAM10000 (e.g., ISIC 2018/2019 task images excluding the HAM10000 split, DermNet, DermQuest, BCN20000, MSK-1/2/3/4 images that are not HAM10000), pretrain a DINOv2 ViT-B/14 with a short JEPA-style or MIM objective for ~20 epochs, then run the EXP-004 recipe on top. Outcomes:
   - **Test AUROC ≥ 0.85** → dermoscopy-domain transfer is sufficient; HAM10000 contamination was not the driver of EXP-007. Strongest paper outcome; the EXP-007 headline survives with the contamination caveat downgraded to "we verified this isn't the cause."
   - **Test AUROC ≈ 0.50–0.80** → partial transfer; some of EXP-007's win was contamination, some was domain.
   - **Test AUROC ≈ 0.30–0.50** → most of EXP-007's win was HAM10000 contamination. Paper headline narrows to "JEPA + frozen backbone unlocked when the backbone has seen the eval dataset; cleanly out-of-dataset frozen backbones don't."
2. **Alternative dermoscopy backbones (MONET, PanDerm raw SSL).** Cheaper than EXP-009 but suffers the same contamination concern as DermLIP since both use ISIC sources. Run only if EXP-009 is blocked.
3. **MedSigLIP-448 swap.** Different architecture (SigLIP-2), different input size (448), trained on SCIN + PAD-UFES-20 + medical mix. Adds another point on the BiomedCLIP-side of the partition but does not directly address the contamination question.
4. **Seed sweep on EXP-007 + EXP-008 configs.** 3–5 seeds each. Cheap, confirms 0.945 and 0.325 are not single-seed artifacts. Do before paper draft, after EXP-009.

Explicitly still not yet:

- **Backbone unfreezing.** Hold until the contamination partition is settled. Unfreezing on top of contaminated weights is much less informative than on top of cleanly-pretrained weights.
- **Larger / multi-class evaluations.** The arc's contribution is now sharply defined; broadening would dilute.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-biomedclip-exp008-v1 \
  ./scripts/hf_jobs_ham10000_exp008.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-biomedclip-exp008-v1
```

Expected top line: `auroc: 0.3247`, `strongest_baseline: pixel_l2 = 0.5802`, `delta_vs_baseline: −0.2556`, `collapsed: False`, `tier: public`.

### 8.3 Config diff vs EXP-007

```diff
 embeddings:
   models:
-    - model_id: dermlip_b16
+    - model_id: biomedclip_b16
       kind: open_clip
-      model_name: hf-hub:redlessone/DermLIP_ViT-B-16
+      model_name: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
       batch_size: 16
       device: auto

 training:
-  model_id: jepa_predictor_ham10000_exp007_v1
-  embedding_model_id: dermlip_b16
+  model_id: jepa_predictor_ham10000_exp008_v1
+  embedding_model_id: biomedclip_b16
```

No code changes vs EXP-007. The `open_clip` kind added in `ba3afce` covers BiomedCLIP via the same `hf-hub:` prefix loader path.

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Replacing DermLIP with BiomedCLIP — same OpenAI CLIP ViT-B/16 architecture, CLIP-trained on PMC-15M (15M PubMed Central biomedical figure-caption pairs, MIT-licensed, no HAM10000) instead of Derm1M dermatology pairs — drops test AUROC on the EXP-004 `strong_held_out_2` proxy from **0.945 to 0.325** [0.303, 0.349]. BiomedCLIP lifts test AUROC over OpenAI CLIP by only +0.04 (0.286 → 0.325); DermLIP lifts over BiomedCLIP by +0.62 (0.325 → 0.945). The pretraining-data gradient is monotone (web < general medical < dermoscopy) but **15× more lift comes from the dermoscopy-specific step than from the general-medical step**. EXP-007's win is therefore localised to dermoscopy-specific pretraining rather than to medical-domain pretraining in general; EXP-008 cannot further partition "dermoscopy-domain transfer" from "HAM10000 image-level overlap."

### 9.2 Numbers safe to quote

- BiomedCLIP-linear train AUROC: **0.923** [0.911, 0.934]
- BiomedCLIP-linear val AUROC: 0.351 [0.327, 0.375]
- BiomedCLIP-linear test AUROC: **0.325** [0.303, 0.349]
- Raw BiomedCLIP cosine baseline: **0.047** [0.040, 0.055] (extreme inversion, comparable to OpenAI CLIP at 0.036)
- Pixel L2 baseline: 0.580 (unchanged)
- Delta vs strongest baseline: **−0.256**
- Linear-predictor lift over raw cosine: **+0.278** (0.047 → 0.325)
- Lift vs OpenAI CLIP linear (EXP-006b): **+0.04** AUROC
- Drop vs DermLIP linear (EXP-007): **−0.62** AUROC
- Train → test drop: **−0.60** (similar to EXP-004's −0.65, very different from EXP-007's −0.05)

### 9.3 Pedagogical beats

1. **"Domain-aligned ≠ task-aligned."** General biomedical pretraining is "domain-aligned" with a dermoscopy proxy in the loose sense (medical images, learned via image-text alignment). It buys +0.04 AUROC. The +0.62 jump comes from pretraining-data that is *task-specifically* aligned (dermoscopy itself). Domain alignment is not a binary; the relevant axis is task-specific corpus structure.
2. **"A monotone gradient with 15× non-uniformity tells you where the work is happening."** The web → BiomedCLIP step adds 0.04. The BiomedCLIP → DermLIP step adds 0.62. Reading the gradient localises causation to the dermoscopy-specific step without any further experiments.
3. **"Negative results inside a positive arc are valuable."** EXP-008 lands at 0.325, which is "still inverted." It would be tempting to read it as "another failure"; the right reading is "useful partition." Negative results that disambiguate prior positive results are first-class evidence.
4. **"Contamination caveats are bigger than they look."** EXP-007 §6 flagged HAM10000 contamination as the central limitation. EXP-008's result strongly elevates that flag: now we know general-medical pretraining doesn't solve the proxy, so DermLIP's win must come from either dermoscopy-specific transfer or HAM10000 overlap, and we cannot tell which from existing experiments.

### 9.4 Updated cross-run table

| Run | Backbone | Pretraining | Predictor | Optimizer | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | +0.269 |
| EXP-003 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | LVD-142M (web) | MLP (underfit) | SGD | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| EXP-006a | DINOv2 B/14 | LVD-142M (web) | MLP (fit) | Adam | 0.893 / 0.266 / 0.248 | Pixel L2 = 0.580 | −0.332 |
| EXP-006b | OpenAI CLIP B/16 | LAION (web) | linear | SGD | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |
| EXP-007 | DermLIP B/16 | Derm1M (dermatology) | linear | SGD | 0.9999 / 0.944 / 0.945 | Pixel L2 = 0.580 | **+0.364** |
| **EXP-008** | **BiomedCLIP B/16** | **PMC-15M (biomedical)** | **linear** | **SGD** | **0.923 / 0.351 / 0.325** | Pixel L2 = 0.580 | **−0.256** |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-008 |
|---|---|
| Domain-distance partition | §2.1, §4.1, §4.3 |
| Three-way pretraining-data ablation | §4.3 (the central table) |
| Limitations of the EXP-007 win | §5.1, §5.3 |
| Updated thesis statement | §5.2 |
| EXP-009 design | §7 |
| Pedagogical insight on domain vs task alignment | §9.3 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — pixel L2, SSIM, raw BiomedCLIP cosine on `strong_held_out_2`. Raw BiomedCLIP cosine histogram is visually inverted, similar to OpenAI CLIP's in EXP-006b.
- `artifacts/plots/jepa_score_histogram.png` — BiomedCLIP-linear test-split JEPA scores. Stable/changing distributions weakly inverted, much less extreme than EXP-006b but visibly worse than EXP-007.
- A side-by-side panel of EXP-006b / EXP-008 / EXP-007 JEPA score histograms would make the gradient visually immediate; recommended for any paper figure.

### 9.7 Nine-experiment arc update

The arc now reads as a complete characterisation, an apparent fix, and a partition that re-narrows the claim:

1. **EXP-001 → EXP-006a/b**: characterise the failure of frozen natural-image backbones across two architectures and three scaffolds.
2. **EXP-007**: dermoscopy-pretrained DermLIP unlocks the proxy at AUROC 0.945. Apparent positive result with contamination caveat front-loaded.
3. **EXP-008**: general-medical-pretrained BiomedCLIP barely lifts AUROC (0.325, +0.04 over web). The win in EXP-007 is concentrated at the dermoscopy-specific pretraining step, not at "medical-domain pretraining" broadly. The EXP-007 contamination caveat is now load-bearing for the paper claim.

Narrative-level: the project has converged on a precise claim with a precisely-scoped follow-up. The next experiment partitions dermoscopy-domain transfer from HAM10000 contamination; after that the paper has a complete and honest result.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-27 | AbdelStark + Claude Code | Initial report. Run completed on commit `c741632`. Three-way pretraining-data ablation (web / general-medical / dermoscopy) now closed; gradient confirmed monotone but 15× non-uniform. EXP-007 paper claim narrowed from "domain-aligned medical pretraining unlocks JEPA" to "dermoscopy-specific pretraining unlocks JEPA, with HAM10000 contamination unpartitioned." EXP-009 scope locked to a non-HAM10000 dermoscopy SSL pretrain. |
