# EXP-008 — HAM10000 BiomedCLIP backbone partition for EXP-007 contamination caveat (`ham10000-hf-biomedclip-exp008-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Partial generalisation of EXP-007's win. Test AUROC on `strong_held_out_2` under frozen BiomedCLIP (PMC-15M, broadly biomedical, did not see HAM10000) is **0.325** [0.303, 0.349] — between OpenAI CLIP's 0.286 (EXP-006b, web-pretrained, did not see HAM10000) and DermLIP's 0.945 (EXP-007, Derm1M, almost certainly contains HAM10000), much closer to CLIP than to DermLIP. The +0.04 AUROC lift from "web pretraining" to "general biomedical pretraining" is small; the +0.62 AUROC jump from "general biomedical" to "dermoscopy-specific" carries almost the entire EXP-007 win. EXP-008 narrows the EXP-007 claim from "domain-aligned medical pretraining unlocks JEPA" to "dermoscopy-specific pretraining unlocks JEPA," and EXP-008 alone cannot separate that from "pretraining that saw HAM10000."
**Date (UTC):** 2026-04-27
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-biomedclip-exp008-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-biomedclip-exp008-v1/`
**Launcher commit:** `c741632`

> **Correction (2026-06-05).** Where this report says Derm1M "almost certainly
> contains" HAM10000, read "confirmed to contain HAM10000 images via literature
> scraping, at small (lower-bounded) scale; not a named source." HAM10000 is
> **not** a named Derm1M image source (its named public datasets are SCIN and
> MSKCC) nor in Derm1M's ISIC-labeled partition, but a perceptual-hash audit
> (2026-06-05, [`docs/audits/derm1m-ham10000-overlap-audit.md`](../audits/derm1m-ham10000-overlap-audit.md))
> confirmed ≥13 distinct HAM10000 images present via reproduced PubMed figures.
> The EXP-008 partition logic is unaffected: BiomedCLIP (PMC-15M) still rules out
> "any general-medical pretraining suffices" without resolving dermoscopy-domain
> transfer versus HAM10000 overlap. See the paper's Appendix H.

---

## 1. Summary

EXP-007 reported test AUROC 0.945 on `strong_held_out_2` after swapping the frozen backbone from OpenAI CLIP ViT-B/16 to DermLIP_ViT-B-16 (same architecture, CLIP-trained on Derm1M dermatology image-text pairs). Two competing readings of that result remained open:

- **Domain-aligned-medical-pretraining hypothesis:** any vision encoder pretrained on medical / clinical images would unlock the proxy. DermLIP's win is representative of a broader class.
- **HAM10000-contamination hypothesis:** Derm1M almost certainly contains HAM10000, so DermLIP has effectively been pretrained on the image distribution the proxy evaluates on. The win is dataset-specific rather than domain-general.

EXP-008 runs the same EXP-004 recipe under `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` — OpenAI CLIP ViT-B/16 architecture, MIT-licensed, CLIP-trained on PMC-15M (~15M PubMed Central figure-caption pairs spanning radiology, pathology, microscopy, anatomy, and a long tail of clinical figures). PMC-15M is broadly biomedical but contains very little dermoscopy and does not include HAM10000 / ISIC archive. It is the cleanest publicly available "domain-aligned medical pretraining without HAM10000 contamination" backbone.

Headline numbers:

- Test AUROC **0.325** [0.303, 0.349]. Above OpenAI CLIP (0.286) by +0.04. Below DermLIP (0.945) by −0.62.
- Delta vs strongest baseline: **−0.256**. Still inverted, still below pixel L2.
- Train AUROC: **0.923** [0.911, 0.934]. Tighter than DINOv2 linear (EXP-004 0.900), looser than OpenAI CLIP linear (EXP-006b 0.986) and DermLIP linear (EXP-007 0.9999).
- Train→test drop: **−0.60 AUROC**, near EXP-004's −0.65. No clean generalisation to `strong_held_out_2` under BiomedCLIP.
- Raw BiomedCLIP cosine baseline: **0.047** [0.040, 0.055]. Inverted, similar magnitude to OpenAI CLIP's 0.036 and unlike DermLIP's 0.109.

Three-backbone comparison:

| Backbone | Pretraining | HAM10000 in pretrain? | Test AUROC | Δ vs preceding |
|---|---|---|---:|---:|
| OpenAI CLIP ViT-B/16 (EXP-006b) | OpenAI WIT (web image-text) | No | 0.286 | — |
| **BiomedCLIP ViT-B/16 (EXP-008)** | **PMC-15M biomedical figures** | **No** | **0.325** | **+0.04** |
| DermLIP ViT-B/16 (EXP-007) | Derm1M dermatology | Likely yes | 0.945 | +0.62 |

The gradient is monotone (web < general medical < dermoscopy in test AUROC) but non-uniform: the web → general-medical step adds 0.04 AUROC, the general-medical → dermoscopy step adds 0.62 AUROC. Roughly 15× more lift from the dermoscopy-specific step than from the medical-domain step. EXP-008 alone cannot further partition "dermoscopy-specific transfer" from "HAM10000 image-level overlap"; that requires EXP-009.

Collapse checks pass. Lesion-ID leakage probes return zero.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-007

One knob only. Pretraining-data axis. Architecture, scaffold, optimiser, splits, nuisance layout, preprocessing, and every other knob byte-identical to EXP-007 (which is itself byte-identical to EXP-006b modulo backbone).

| Knob | EXP-007 | EXP-008 |
|---|---|---|
| `embeddings.models[0].model_name` | `hf-hub:redlessone/DermLIP_ViT-B-16` | **`hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`** |
| `training.embedding_model_id` | `dermlip_b16` | **`biomedclip_b16`** |
| Architecture | OpenAI CLIP ViT-B/16 | OpenAI CLIP ViT-B/16 |
| Embedding dim | 512 | 512 |
| `kind` | `open_clip` | `open_clip` (BiomedCLIP loads as a `CustomTextCLIP` under the same loader path) |
| Predictor / hidden / epochs / batch / LR / weight decay / optimiser | linear / — / 200 / 128 / 0.03 / 0.001 / SGD | same |

Both DermLIP and BiomedCLIP load via `open_clip.create_model_and_transforms('hf-hub:...')` and produce a 512-d projected image embedding via `encode_image`. Image normalisation in both cases follows OpenAI CLIP mean/std. Preprocessing pipeline upstream is the EXP-004 224×224 center-crop profile.

### 2.2 What BiomedCLIP saw during pretraining

PMC-15M is a corpus of ~15M figure-caption pairs scraped from PubMed Central open-access biomedical articles. Per the BiomedCLIP paper, it spans:

- Radiology (CT, MRI, ultrasound, X-ray): substantial fraction.
- Pathology (microscopy, histology, IHC): substantial fraction.
- Surgical, anatomical, and procedural figures: smaller fraction.
- Dermatology (clinical / dermoscopy): small fraction. The corpus contains figures referencing HAM10000 / ISIC datasets in research papers but does not contain the raw datasets themselves.
- Patient-figure mixtures, charts, diagrams: long tail.

The relevant qualitative property for EXP-008's purpose: PMC-15M was not constructed by ingesting HAM10000 or the ISIC archive. Any HAM10000 image content reaching BiomedCLIP would have to have been downsampled, cropped, or otherwise processed for a published figure, and would represent a tiny fraction (likely < 0.01 %) of the pretraining data. For practical purposes, BiomedCLIP did not see HAM10000.

### 2.3 Held constant

Same as EXP-004/005/006a/006b/007: dataset `abdelstark/ham10000`, seed 20260422, lesion-ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs per split, strict same-diagnosis-site changing-pair policy, bootstrap CI protocol (1,000 samples, 95 % CI). Train stable pairs rotate between `strong` and `strong_held_out`; val and test use `strong_held_out_2`.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. Single attempt.

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
| Raw BiomedCLIP cosine | 0.047 | [0.040, 0.055] | stable ≫ changing (inverted) |

The predictor lifts raw BiomedCLIP cosine from 0.047 → 0.325 (+0.28 AUROC) — comparable in magnitude to the recovery EXP-006b's predictor did over OpenAI CLIP (0.036 → 0.286, +0.25). The ceiling is still below random. BiomedCLIP looks structurally more like OpenAI CLIP than like DermLIP from this proxy's perspective despite being trained on a broadly medical corpus.

### 4.2 JEPA across splits

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.923** | [0.911, 0.934] |
| val | `strong_held_out_2` | 0.351 | [0.327, 0.375] |
| test | `strong_held_out_2` | **0.325** | [0.303, 0.349] |

Train-to-test drop is −0.60 AUROC. Compare to EXP-007's −0.05 (clean generalisation) and EXP-006b's −0.70 (catastrophic generalisation failure). BiomedCLIP is closer to the failure mode than to the success mode: in-distribution fit is solid (train 0.92) but does not transfer to the unseen nuisance family.

### 4.3 The three-way pretraining-data comparison

| Run | Backbone | Pretraining corpus | Train AUROC | Val AUROC | Test AUROC | Δ vs strongest |
|---|---|---|---:|---:|---:|---:|
| EXP-006b | OpenAI CLIP B/16 | OpenAI WIT (≈400M web image-text pairs) | 0.986 | 0.300 | 0.286 | −0.294 |
| **EXP-008** | **BiomedCLIP B/16** | **PMC-15M (biomedical figures)** | **0.923** | **0.351** | **0.325** | **−0.256** |
| EXP-007 | DermLIP B/16 | Derm1M (dermatology) | 0.9999 | 0.944 | 0.945 | +0.364 |

Three points worth flagging:

- The gradient is monotone but non-uniform. OpenAI CLIP < BiomedCLIP < DermLIP on test AUROC, in the order of "domain-distance to dermoscopy." The gap from CLIP to BiomedCLIP is +0.04; the gap from BiomedCLIP to DermLIP is +0.62. The dermoscopy-specific step carries the bulk of the win.
- Train AUROCs do not predict test AUROCs across this row. OpenAI CLIP fits train hardest (0.986) and lands at 0.286. DermLIP fits train at 0.9999 and lands at 0.945. BiomedCLIP fits train less hard than either (0.92) and lands in between. The relationship between train fit and test fit is mediated by the embedding space's structural invariance to nuisance, not by raw representational capacity.
- The mechanistic claim from EXP-007 §4.4 is partially confirmed and partially complicated. EXP-007 hypothesised that DermLIP organises nuisance directions consistently across families. EXP-008 shows BiomedCLIP does not organise them this way despite being a medical-domain encoder. Domain-aligned pretraining alone is insufficient; what matters is whether the corpus contains enough on-task structure to align nuisance directions across families. For dermoscopy, that threshold is somewhere between PMC-15M (insufficient) and Derm1M (sufficient).

### 4.4 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000769 | 0.000946 |
| 100 | 0.000557 | 0.001041 |
| 200 | 0.000557 | 0.001041 |

Train loss drops 28 % over the first 100 epochs and flat-lines. Val loss rises 10 % and flat-lines. The shape of the curve is nearly identical to EXP-006b (CLIP linear): the predictor fits the training distribution efficiently, val loss diverges, val AUROC stays well below random. Same overall pattern as the natural-image-backbone runs, with marginally more lift on the test side.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.99999988
- `dimension_variance_mean` = 2.65 × 10⁻⁴
- `dimension_variance_min` = 5.83 × 10⁻⁵
- `collapsed` = **False**

`dimension_variance_min` is the highest of any frozen-natural-image-backbone run on the EXP-004 proxy (EXP-006b 1.4 × 10⁻⁵; EXP-006a 2.5 × 10⁻⁶) but well below EXP-007's DermLIP run on the same metric. Consistent with BiomedCLIP's embedding space having slightly more usable structure than OpenAI CLIP's, but much less than DermLIP's.

### 4.6 JEPA pair-score distributions across splits

Scores are per-pair predicted-target distances; lower distance = correct for stable pairs. Gap = `stable_mean − changing_mean`; negative = correct (stable closer), positive = inverted.

| Split | Stable mean | Changing mean | Gap |
|---|---:|---:|---:|
| train | 0.151 | 0.259 | **−0.108 (correct)** |
| val | 0.285 | 0.259 | +0.026 (inverted, weak) |
| test | 0.286 | 0.253 | **+0.033 (inverted, weak)** |

Train gap is the right sign and meaningful magnitude. Val and test gaps flip to inverted but with very small magnitude (~0.03), compared to OpenAI CLIP's (+0.05 test) and DINOv2's (+0.10–0.12). BiomedCLIP's inverted gap on test is the smallest of any frozen-natural-or-general-medical-backbone run on this proxy, but still on the wrong side of zero, so test AUROC stays below 0.5.

---

## 5. Analysis

### 5.1 What EXP-008 actually proved

**Proved:** A frozen vision backbone with general biomedical pretraining (BiomedCLIP, PMC-15M) produces test AUROC 0.325 on the EXP-004 `strong_held_out_2` proxy — slightly better than web-pretrained OpenAI CLIP (0.286), and well below dermoscopy-pretrained DermLIP (0.945). The lift from web to general biomedical is small (+0.04); the lift from general biomedical to dermoscopy-specific is large (+0.62).

**Proved:** EXP-007's "domain-aligned pretraining unlocks JEPA" claim does not generalise to "any medical-image pretraining." The win is concentrated at the dermoscopy-specific step.

**Not proved:** Whether the dermoscopy-specific step's win is caused by (a) HAM10000 being part of Derm1M during DermLIP pretraining, or (b) the broader dermoscopy-domain visual structure in Derm1M being sufficient even without HAM10000. EXP-008 closes the "general medical" alternative but cannot disambiguate (a) from (b). A non-HAM10000 dermoscopy-pretrained backbone is needed; this is harder to obtain because every public dermoscopy SSL pretrain known (PanDerm, MONET, DermLIP, …) uses ISIC archives that include HAM10000 as a component.

### 5.2 Updated thesis-level statement after nine runs

> On a leakage-controlled HAM10000 longitudinal-proxy task with three disjoint nuisance families and evaluation on the third (unseen) family, a JEPA-style linear predictor over a frozen vision backbone:
>
> - Wins on matched-eval (DINOv2 ViT-B/14 linear, +0.27 AUROC, EXP-002).
> - Loses on one held-out family (−0.28, EXP-003).
> - Inverts below random on a third unseen family (−0.33, EXP-004), and that inversion is preserved under MLP scaffolds (EXP-005, EXP-006a) and a non-medical backbone swap (OpenAI CLIP, EXP-006b, −0.29).
> - Reaches +0.36 AUROC over baselines when the backbone is DermLIP, dermoscopy-CLIP-trained on Derm1M which almost certainly includes HAM10000 (EXP-007).
> - Lifts only marginally (+0.04 over OpenAI CLIP) when the backbone is BiomedCLIP, broadly biomedical-CLIP-trained on PMC-15M which does not include HAM10000 (EXP-008). General-medical pretraining is insufficient.
>
> The nine-run sequence localises the success regime to a narrow band: dermoscopy-specific pretraining is required, and a non-trivial fraction of that requirement may be HAM10000 image-level overlap rather than dermoscopy-domain transfer in general. The defensible claim is "frozen DermLIP + linear JEPA predictor solves this proxy; general-medical or natural-image frozen backbones do not; the specificity of 'dermoscopy-pretrained' versus 'HAM10000-pretrained' is not yet partitioned."

### 5.3 What EXP-008 tells us about the EXP-007 mechanism

EXP-007 §4.4 hypothesised that DermLIP's embedding space has consistent nuisance directions across all three nuisance families, which is what allows a linear predictor trained on the first two families to generalise to the third. EXP-008 sharpens that hypothesis by ruling out one alternative:

- **Ruled out:** "Any embedding space trained on a corpus that contains medical images" has consistent nuisance directions. BiomedCLIP saw 15× more images than DermLIP did, including a non-trivial fraction of dermatology figures, and does not exhibit cross-family direction consistency.
- **Still open:** Whether the consistency in DermLIP comes from (a) the dermoscopy-specific visual statistics in Derm1M's non-HAM10000 portion or (b) the HAM10000-specific image content. A non-HAM10000 dermoscopy SSL run, compared to DermLIP on this proxy, would partition (a) from (b). EXP-009 should target this.

---

## 6. Limitations and threats to validity

1. **EXP-008 cannot disambiguate dermoscopy-domain transfer from HAM10000 contamination.** Central limitation. The cleanest follow-up requires a non-HAM10000 dermoscopy pretrain.
2. **One general-medical backbone.** BiomedCLIP is one specific point in the medical-pretraining design space. MedSigLIP-448 (Google), MONET (UW / Allen AI), or other BioMed-CLIP variants might land at different AUROCs.
3. **Architecture match still ViT-B/16 OpenAI-CLIP-style.** The three-way table holds architecture constant, which is the right choice for an ablation, but means the comparison cannot speak to whether other architectures (DINOv2-style self-distillation, transformer-only image backbones) would behave differently.
4. **Single seed.** Across all nine runs.
5. **One third-family design.** `strong_held_out_2` is one specific synthetic-augmentation family.
6. **HAM10000 is cross-sectional.** Longitudinal caveats unchanged.

---

## 7. What changes for the next run (EXP-009 scoping)

Partitioning "dermoscopy-domain transfer" from "HAM10000 contamination" is the decision-relevant next move. In priority order:

1. **EXP-009 — Self-pretrain DINOv2 on a non-HAM10000 dermoscopy corpus.** Build a corpus of dermoscopic / clinical skin images that explicitly excludes HAM10000 (e.g., ISIC 2018/2019 task images excluding the HAM10000 split, DermNet, DermQuest, BCN20000, MSK-1/2/3/4 images that are not HAM10000), pretrain a DINOv2 ViT-B/14 with a short JEPA-style or MIM objective for ~20 epochs, then run the EXP-004 recipe on top. Outcomes:
   - Test AUROC ≥ 0.85 → dermoscopy-domain transfer is sufficient; HAM10000 contamination was not the driver of EXP-007.
   - Test AUROC ≈ 0.50–0.80 → partial transfer; some of EXP-007's win was contamination, some was domain.
   - Test AUROC ≈ 0.30–0.50 → most of EXP-007's win was HAM10000 contamination; the headline narrows to "JEPA + frozen backbone unlocks when the backbone has seen the eval dataset."
2. **Alternative dermoscopy backbones (MONET, PanDerm raw SSL).** Cheaper than EXP-009 but suffers the same contamination concern as DermLIP since both use ISIC sources. Run only if EXP-009 is blocked.
3. **MedSigLIP-448 swap.** Different architecture (SigLIP-2), different input size (448), trained on SCIN + PAD-UFES-20 + medical mix. Adds another point on the BiomedCLIP-side of the partition but does not directly address contamination.
4. **Seed sweep on EXP-007 + EXP-008 configs.** 3–5 seeds each. Cheap; confirms 0.945 and 0.325 are not single-seed artifacts.

Explicitly still not yet:

- Backbone unfreezing. Hold until the contamination partition is settled. Unfreezing on top of contaminated weights is much less informative than on top of cleanly-pretrained weights.
- Larger / multi-class evaluations. The contribution is the failure / success characterisation; broadening dilutes it.

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

> Replacing DermLIP with BiomedCLIP — same OpenAI CLIP ViT-B/16 architecture, CLIP-trained on PMC-15M (15M PubMed Central biomedical figure-caption pairs, MIT-licensed, no HAM10000) instead of Derm1M dermatology pairs — drops test AUROC on the EXP-004 `strong_held_out_2` proxy from 0.945 to 0.325 [0.303, 0.349]. BiomedCLIP lifts test AUROC over OpenAI CLIP by only +0.04 (0.286 → 0.325); DermLIP lifts over BiomedCLIP by +0.62 (0.325 → 0.945). The pretraining-data gradient is monotone (web < general medical < dermoscopy) but roughly 15× more lift comes from the dermoscopy-specific step than from the general-medical step. EXP-007's win is therefore localised to dermoscopy-specific pretraining rather than to medical-domain pretraining in general; EXP-008 cannot further partition "dermoscopy-domain transfer" from "HAM10000 image-level overlap."

### 9.2 Numbers safe to quote

- BiomedCLIP-linear train AUROC: **0.923** [0.911, 0.934]
- BiomedCLIP-linear val AUROC: 0.351 [0.327, 0.375]
- BiomedCLIP-linear test AUROC: **0.325** [0.303, 0.349]
- Raw BiomedCLIP cosine baseline: **0.047** [0.040, 0.055]
- Pixel L2 baseline: 0.580 (unchanged)
- Delta vs strongest baseline: **−0.256**
- Linear-predictor lift over raw cosine: **+0.278** (0.047 → 0.325)
- Lift vs OpenAI CLIP linear (EXP-006b): **+0.04** AUROC
- Drop vs DermLIP linear (EXP-007): **−0.62** AUROC
- Train → test drop: **−0.60**

### 9.3 Pedagogical beats

1. **Domain-aligned ≠ task-aligned.** General biomedical pretraining is "domain-aligned" with a dermoscopy proxy in the loose sense. It buys +0.04 AUROC. The +0.62 jump comes from pretraining-data that is task-specifically aligned (dermoscopy itself). Domain alignment is not a binary; the relevant axis is task-specific corpus structure.
2. **A monotone gradient with non-uniform steps localises causation.** Web → BiomedCLIP adds 0.04. BiomedCLIP → DermLIP adds 0.62. Reading the gradient localises the work to the dermoscopy-specific step without further experiments.
3. **A negative result that disambiguates a prior positive is first-class evidence.** EXP-008 lands at 0.325 ("still inverted"), but the value is in the partition: now we know general-medical pretraining doesn't solve the proxy.
4. **Contamination caveats can become load-bearing.** EXP-007 §6 flagged HAM10000 contamination as the central limitation. EXP-008 elevates it: DermLIP's win must come from either dermoscopy-specific transfer or HAM10000 overlap, and we cannot tell which from existing experiments.

### 9.4 Updated cross-run table

| Run | Backbone | Pretraining | Predictor | Optimizer | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---|---:|---|---:|
| EXP-001 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | +0.269 |
| EXP-003 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | DINOv2 B/14 | LVD-142M (web) | linear | SGD | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| EXP-005 | DINOv2 B/14 | LVD-142M (web) | MLP (underfit) | SGD | 0.572 / 0.293 / 0.270 | Pixel L2 = 0.580 | −0.310 |
| EXP-006a | DINOv2 B/14 | LVD-142M (web) | MLP (fit) | Adam | 0.893 / 0.266 / 0.248 | Pixel L2 = 0.580 | −0.332 |
| EXP-006b | OpenAI CLIP B/16 | WIT (OpenAI web) | linear | SGD | 0.986 / 0.300 / 0.286 | Pixel L2 = 0.580 | −0.294 |
| EXP-007 | DermLIP B/16 | Derm1M (dermatology) | linear | SGD | 0.9999 / 0.944 / 0.945 | Pixel L2 = 0.580 | **+0.364** |
| **EXP-008** | **BiomedCLIP B/16** | **PMC-15M (biomedical)** | **linear** | **SGD** | **0.923 / 0.351 / 0.325** | Pixel L2 = 0.580 | **−0.256** |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-008 |
|---|---|
| Domain-distance partition | §2.1, §4.1, §4.3 |
| Three-way pretraining-data ablation | §4.3 |
| Limitations of the EXP-007 win | §5.1 |
| Updated thesis statement | §5.2 |
| EXP-009 design | §7 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — pixel L2, SSIM, raw BiomedCLIP cosine on `strong_held_out_2`. Raw BiomedCLIP cosine histogram is visibly inverted, similar to OpenAI CLIP's in EXP-006b.
- `artifacts/plots/jepa_score_histogram.png` — BiomedCLIP-linear test-split JEPA scores. Stable/changing distributions weakly inverted, less extreme than EXP-006b but visibly worse than EXP-007.
- A side-by-side panel of EXP-006b / EXP-008 / EXP-007 JEPA score histograms makes the gradient visually immediate; recommended for any paper figure.

### 9.7 Nine-experiment summary

1. **EXP-001 → EXP-006a/b:** characterise the failure of frozen natural-image backbones across two architectures and three scaffolds.
2. **EXP-007:** dermoscopy-pretrained DermLIP unlocks the proxy at AUROC 0.945. Positive result with contamination caveat front-loaded.
3. **EXP-008:** general-medical-pretrained BiomedCLIP barely lifts AUROC (0.325, +0.04 over web). The win in EXP-007 is concentrated at the dermoscopy-specific pretraining step. The contamination caveat is now load-bearing.

EXP-009 partitions dermoscopy-domain transfer from HAM10000 contamination.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-27 | Abdelhamid Bakhta | Initial report. Run completed on commit `c741632`. Three-way pretraining-data ablation (web / general-medical / dermoscopy) closed; gradient monotone but non-uniform. EXP-007 claim narrowed from "domain-aligned medical pretraining unlocks JEPA" to "dermoscopy-specific pretraining unlocks JEPA, with HAM10000 contamination unpartitioned." EXP-009 scope locked to a non-HAM10000 dermoscopy SSL pretrain. |
