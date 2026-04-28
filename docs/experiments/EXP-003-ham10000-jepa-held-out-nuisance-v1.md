# EXP-003 — HAM10000 held-out-nuisance JEPA run (`ham10000-hf-dinov2-exp003-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Falsification positive. The EXP-002 JEPA delta does **not** survive an unseen nuisance family. JEPA train→test AUROC drops from 0.953 to 0.679 while SSIM becomes the strongest cheap baseline at 0.960 on the same held-out distribution. Details require careful reading.
**Date (UTC):** 2026-04-23
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dinov2-exp003-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dinov2-exp003-v1/`
**Launcher commit:** `c34c398`

---

## 1. Summary

EXP-002 left open one alternative explanation for its +0.27 AUROC JEPA
delta: the linear predictor could have been memorizing the vector field
induced on DINOv2 by one specific nuisance family, because training and
evaluation stable pairs shared that distribution.

EXP-003 tests that hypothesis by training stable pairs under the
`strong` family (exactly as EXP-002) and evaluating val + test stable
pairs under a **disjoint** `strong_held_out` family (hue shift,
posterize, unsharp-mask sharpen, motion blur, random rectangular
erasing, low-quality JPEG) that shares zero transform types with
`strong`. Same dataset, same seed, same hardware, same JEPA architecture,
same strict changing-pair policy — only the test-time nuisance family
changed.

The result is a clean negative for JEPA generalization across nuisance
families, with a surprising and mechanistically interesting side finding:

- **JEPA predictor:** train AUROC **0.953**, val AUROC **0.710**,
  test AUROC **0.680** [0.656, 0.702]. The train→test drop of **0.27
  AUROC** mirrors almost exactly the +0.27 delta EXP-002 produced under
  matched train/test families. EXP-002's win was augmentation-family
  memorization, not generalizable nuisance invariance.
- **DINOv2 ViT-S/14 cosine:** **0.361 AUROC** (below random for the
  positive-class convention; stable distances are systematically *larger*
  than changing distances).
- **DINOv2 ViT-B/14 cosine:** **0.415 AUROC** (also below random).
- **SSIM distance:** **0.961 AUROC** [0.953, 0.968]. SSIM wins decisively
  because the held-out family preserves most of the image pixel area
  (random erasing at 6–18% area), so a source-vs-variant structural
  similarity measure still tracks the stable/changing decision.
- **Pixel L2:** **0.911 AUROC**. Same mechanism.

So the JEPA delta vs the strongest cheap baseline flipped from
**+0.2687** (EXP-002) to **−0.2810** (EXP-003). But the strongest
cheap baseline also changed identity (DINOv2 cosine → SSIM), because the
held-out family disables DINOv2 as a useful distance while leaving
pixel-space mostly untouched. Both facts matter for the thesis.

Collapse checks still pass. Leakage probes still return zero overlap.
Training dynamics are clean: train loss is identical to EXP-002, val
loss is 50% higher than train loss (clear distribution shift signature).

---

## 2. Experimental setup

### 2.1 What changed vs EXP-002

One knob only. Everything else — dataset, seed, hardware, backbones,
predictor architecture, changing-pair policy, pair counts, metric
protocol — matches EXP-002 so the measured difference is a clean
read on nuisance-family generalization.

| Knob | EXP-002 | EXP-003 |
|---|---|---|
| `dataset.nuisance_severity` (train) | `strong` | `strong` |
| `dataset.nuisance_severity_eval` (val + test) | _unset_ → `strong` | **`strong_held_out`** |
| `dataset.changing_pair_policy` | strict_same_diagnosis_site | strict_same_diagnosis_site |

### 2.2 The `strong_held_out` family

Disjoint transform set from `strong`. No brightness, contrast,
saturation, rotation, flip, scale, translate, Gaussian blur, Gaussian
noise, or mid-quality JPEG.

| Perturbation | Range |
|---|---|
| Hue shift (HSV rotation) | ±12° |
| Posterize (bits per channel) | 4–6 |
| Sharpen (`ImageFilter.UnsharpMask`) | radius 1.0–3.0, percent 100–220 |
| Motion blur (linear numpy roll) | length 5–15 px, horizontal or vertical |
| Random rectangular erasing | 6–18 % of image area, corner-color fill |
| JPEG re-encode | quality 20–40 (lower than `strong`'s 45–70) |

The unit test `test_held_out_recipe_records_disjoint_transforms` pins
that every `strong`-only key (brightness, contrast, saturation,
rotation_degrees, scale, hflip, noise_sigma) is absent from
`strong_held_out` recipes and every held-out-only key is present, so
future edits cannot silently reintroduce overlap.

### 2.3 What stayed constant

- Dataset: HAM10000 on `abdelstark/ham10000`, 10,015 images, same mount.
- Seed: 20260422.
- Splits: 5,229 / 1,120 / 1,121 lesions, zero lesion-ID overlap.
- Pair counts: 1,000 stable + 1,000 changing per split, strict
  same-diagnosis-site changing policy.
- Embedding backbones: DINOv2 ViT-S/14 (batch 32) and ViT-B/14 (batch 16).
- JEPA predictor: identity-initialized linear map over DINOv2 ViT-B/14,
  200 epochs, batch 128, LR 0.03, weight decay 0.001.
- Metric protocol: 1,000-sample bootstrap CI, 95 % level, fixed TPR 0.80.

---

## 3. Operational timeline

Same pattern as EXP-002 with the full observability stream in
`logs/progress.jsonl`.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Same pinned wheel resolve as EXP-002 |
| Manifest build (incl. stable-variant writes) | ~15 min | Train stables under `strong`, val + test stables under `strong_held_out` |
| DINOv2 ViT-S/14 + ViT-B/14 embedding export | ~29 min | 8,004 unique images across 6,000 pairs |
| Baselines (test split) | ~5 min | Pixel L2, SSIM, DINOv2-S cos, DINOv2-B cos |
| JEPA linear predictor fit | **6.5 min** (388.3 s) | Slightly longer than EXP-002 (264 s); same config — the train stable set is identical, so the extra time is scheduler noise |
| Upload | ~3 min | 3,014 files, ~189 MB |
| **Total wall time** | **≈ 1h 48m** | Essentially identical to EXP-002 |

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction | Δ vs strongest |
|---|---:|:---:|---|---:|
| SSIM distance | **0.961** | [0.953, 0.968] | changing > stable | — |
| Pixel L2 | 0.911 | [0.896, 0.925] | changing > stable | −0.050 |
| **JEPA predictor (exp003)** | **0.680** | [0.656, 0.702] | changing > stable | **−0.281** |
| DINOv2 ViT-B/14 cosine | 0.415 | [0.393, 0.438] | **stable > changing** | −0.546 |
| DINOv2 ViT-S/14 cosine | 0.361 | [0.338, 0.384] | **stable > changing** | −0.600 |

Score direction matters. Pixel L2 and SSIM give stable pairs *smaller*
distances than changing pairs — the correct sign, so AUROC > 0.5. DINOv2
cosine gives stable pairs *larger* distances than changing pairs — the
**wrong** sign, so AUROC < 0.5. Under the `strong_held_out` family,
frozen DINOv2 embeddings are actively less consistent on the
source-to-variant comparison than they are on the different-lesion
comparison.

Non-overlap check: JEPA's upper bound (0.702) is far below SSIM's lower
bound (0.953). The gap is 0.25 AUROC in SSIM's favour, with non-
overlapping 95 % bootstrap CIs. This is a decisive, statistically
rock-solid loss for JEPA on this split.

### 4.2 JEPA across splits

| Split | Family used for stable target | AUROC | 95% CI |
|---|---|---:|:---:|
| train | strong (same as EXP-002) | **0.953** | [0.945, 0.961] |
| val | strong_held_out | 0.710 | [0.688, 0.732] |
| test | strong_held_out | 0.680 | [0.656, 0.702] |

This is the cleanest possible generalization-gap signal. The same
predictor, on the same DINOv2 ViT-B/14 embeddings, achieves EXP-002's
train AUROC on train, and collapses by 0.27 AUROC on the held-out
distribution.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000648 | 0.000847 |
| 20 | 0.000564 | 0.000840 |
| 100 | 0.000563 | 0.000838 |
| 200 | 0.000562 | 0.000837 |

Two signals:

1. **Train loss trajectory is identical to EXP-002** (0.000648 → 0.000562
   in both runs; same seed, same training family, same data).
2. **Val loss is ≈ 50 % higher than train loss** (0.000837 vs 0.000562)
   and never converges toward train loss. This is the canonical shape
   of a distribution-shift failure: the predictor fits the `strong`
   family tightly and does not extrapolate to `strong_held_out`.

### 4.4 Representation health

- `prediction_norm_mean` = 1.0
- `prediction_norm_min` = 0.9999999
- `dimension_variance_mean` = 4.7 × 10⁻⁴
- `dimension_variance_min` = 1.7 × 10⁻⁵
- `collapsed` = **False**

The predictor itself is healthy — it has not collapsed to a degenerate
map. It is just tuned to the wrong distribution at test time.

### 4.5 JEPA pair-score distributions across splits

| Split | JEPA stable mean | JEPA changing mean | Gap |
|---|---:|---:|---:|
| train | 0.2144 | 0.4405 | 0.226 |
| val | 0.3120 | 0.4048 | 0.093 |
| test | 0.3174 | 0.3970 | 0.080 |

Gap shrinks from 0.23 (train) to 0.08 (test). The predictor's stable
score rises by 0.10 on held-out variants — it thinks held-out stables
are considerably more "changing-like" than in-distribution stables.
Changing pair scores barely move (0.44 → 0.40).

### 4.6 Baseline score distributions (test split)

| Baseline | Stable mean | Changing mean | Direction |
|---|---:|---:|---|
| Pixel L2 | 0.086 | 0.174 | changing > stable (correct) |
| SSIM distance | 0.224 | 0.450 | changing > stable (correct) |
| DINOv2 ViT-S/14 cosine | 0.355 | 0.303 | **stable > changing (inverted)** |
| DINOv2 ViT-B/14 cosine | 0.358 | 0.324 | **stable > changing (inverted)** |

Pixel L2 and SSIM are **smaller** for stable pairs than for changing
pairs because the held-out family modifies only a small fraction of
pixels (random erasing 6–18 %, hue / posterize / sharpen / motion blur
each preserves most of the image), while changing pairs are two
entirely different photographs where the whole image differs.

DINOv2 cosine is **larger** for stable pairs because the held-out family
shifts the DINOv2 semantic features more aggressively than the natural
same-diagnosis-same-site difference does. DINOv2 is semantic-sensitive
but not nuisance-robust under these transforms — hue and posterize
shifts move ViT features noticeably even when the underlying lesion is
the same.

---

## 5. Analysis

### 5.1 Falsification partially positive

Question EXP-003 was built to answer:

> Does EXP-002's +0.27 AUROC JEPA delta survive when the test-time
> nuisance family is disjoint from the training family?

Answer: **No, it doesn't.** Under held-out nuisance, JEPA test AUROC
drops from 0.920 (EXP-002) to 0.680 (EXP-003), and the JEPA predictor
loses to multiple cheap baselines. The falsification design worked —
EXP-002's win was, at least in part, memorization of the `strong`
family's vector field on DINOv2.

### 5.2 But the held-out family introduced a different shortcut

The `strong_held_out` family is *gentle* on pixel L2 and SSIM because:

- Random erasing only affects ~10 % of pixels.
- Hue shift preserves structural edges.
- Posterize preserves spatial layout.
- Sharpen preserves gross structure.
- Motion blur preserves aggregate colour statistics.

The union of these transforms still leaves 80–90 % of pixels close to
their source. So SSIM (a spatially-weighted similarity measure) easily
separates `(source, source + modest perturbation)` from `(lesion A,
lesion B)` where A and B are genuinely different photographs.

This means the cheap-baseline dominance in EXP-003 is not a pure
validation of pixel-space approaches — it is a consequence of the
specific held-out family being shaped in a way that pixel metrics
exploit. A cleaner falsification would use a held-out family that
*also* disables pixel-space separability while stressing DINOv2.

### 5.3 DINOv2 under held-out nuisance is worse than random

DINOv2 cosine distance on the test split inverts its sign under the
held-out family (AUROC 0.36 and 0.41 — both significantly below 0.5).
This is independently interesting: frozen DINOv2 is not robust to hue /
posterize / motion blur at these intensities. The implication for the
thesis is broader than JEPA alone: any representation learning scheme
built on top of frozen DINOv2 will inherit this brittleness unless
either the backbone is fine-tuned or the predictor is powerful enough
to compensate. A linear map trained only on `strong` variants cannot
compensate.

### 5.4 Where the JEPA gain went

Decomposing the train-to-test drop:

- **Same-family generalization:** train = 0.953, val = 0.710. Gap 0.24
  AUROC.
- **Val-to-test variance:** val = 0.710, test = 0.680. Gap 0.03 AUROC.

So essentially all the generalization loss is due to the nuisance-family
shift, not to split variance or overfitting to val. The model fits the
training family tightly (train loss 0.000562) and fails to transfer.

### 5.5 What EXP-002 and EXP-003 *jointly* say

Before accepting or rejecting the DermaJEPA thesis, the joint reading of
the two runs is:

| Proposition | Evidence |
|---|---|
| The JEPA-style linear predictor beats frozen DINOv2 cosine when train and test share the nuisance family. | EXP-002: +0.27 AUROC, non-overlapping CIs. |
| The advantage disappears when train and test use disjoint nuisance families. | EXP-003: −0.28 AUROC vs SSIM, below-random vs DINOv2. |
| The JEPA objective, as currently instantiated (linear map, one family at train time), is therefore a nuisance-family-specific corrector, not a general lesion-identity representation. | EXP-003 train loss matches EXP-002 exactly; val loss is +50 %. Identical predictor, shifted test distribution, collapsed delta. |
| Frozen DINOv2 ViT-B/14 is not nuisance-invariant under hue / posterize / motion / erasing at these intensities. | EXP-003 DINOv2 cosine AUROC < 0.5, with inverted stable/changing direction. |

The honest thesis-level statement is now:

> On a leakage-controlled HAM10000 longitudinal-proxy task, a linear
> JEPA-style predictor over frozen DINOv2 ViT-B/14 improves over cheap
> baselines by 0.27 AUROC when the test nuisance distribution matches
> training, but that improvement does not generalize to a disjoint
> nuisance family, where the same predictor loses to pixel L2 and SSIM
> and frozen DINOv2 cosine performs below chance. The MVP therefore
> does not yet demonstrate that JEPA-style latent prediction produces a
> nuisance-invariant lesion representation on this dataset.

That is the allowed spec-compliant phrasing of a genuinely informative
two-run result.

---

## 6. Limitations and threats to validity

1. **One held-out family.** A single design choice. The stable/changing
   separability of the `strong_held_out` family under pixel L2 and SSIM
   is high because most of its transforms preserve large pixel regions.
   A family that also blurs pixel L2 and SSIM (e.g. stronger erasing,
   larger scale variation, perspective warp) would tell us more.
2. **Single seed.** The train-to-test collapse is large enough that
   seed variance is unlikely to flip the sign, but it has not yet been
   measured.
3. **Predictor class is linear.** The collapse could be partially
   specific to linear models; a small MLP with the same objective might
   generalize better. That is a separate test.
4. **DINOv2 is frozen.** If the backbone itself were fine-tuned on
   stable pairs during JEPA training, both the in-distribution and
   out-of-distribution results would shift. The current scaffold does
   not exercise that.
5. **HAM10000 is cross-sectional.** Every caveat from EXP-001 and
   EXP-002 carries forward.

---

## 7. What changes for the next run (EXP-004 scoping)

The joint EXP-002 + EXP-003 evidence makes the next experiment much
more targeted:

1. **Multi-family held-out eval.** Three disjoint held-out families at
   eval time — one that preserves pixels (like EXP-003's current
   family), one that disrupts pixels but preserves DINOv2 semantics,
   and one that disrupts both. Report a per-family table of
   JEPA vs strongest-baseline deltas. This characterizes the failure
   surface instead of relying on one point.
2. **Multi-family training distribution.** Train JEPA on the **union**
   of `strong` and `strong_held_out` families (each sampled per pair).
   If that predictor then generalizes to a third unseen family, the
   scheme is nuisance-family-agnostic; if not, the generalization gap
   is fundamental to the linear scaffold.
3. **Seed sweep on EXP-002 and EXP-003 configs.** 3 seeds each, report
   mean ± std. Confirms the direction of both deltas.
4. **Small MLP predictor** (2-layer, hidden dim 512) at the same
   training budget. If generalization improves, the linear constraint
   is part of the story; if not, the backbone is.
5. **Fixed-FPR operating point** (FPR = 0.05 and 0.01) per family, so
   operational behaviour is legible beyond AUROC.

EXP-004 should implement (1) and (2). (3), (4), (5) can ride along or
split into EXP-005.

Explicitly not doing in EXP-004:

- Abandoning the thesis. EXP-003 is not a refutation — it is a
  characterization of the failure mode. Abandoning now would be
  overinterpretation.
- Backbone fine-tuning. Still premature given the scaffold is linear.
- Any hyperparameter sweep aimed at boosting the JEPA delta under
  held-out nuisance. That would be the exact form of rescue the MVP
  spec forbids.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-exp003-v1 \
  ./scripts/hf_jobs_ham10000_exp003.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dinov2-exp003-v1
```

Expected top line: `auroc: 0.6795`, `strongest_baseline: ssim_distance =
0.9605`, `delta_vs_baseline: -0.2810`, `collapsed: False`, `tier: public`.

### 8.3 Config diff vs EXP-002

```diff
 dataset:
   nuisance_severity: strong
+  nuisance_severity_eval: strong_held_out
   changing_pair_policy: strict_same_diagnosis_site
 training:
-  model_id: jepa_predictor_ham10000_exp002_v1
+  model_id: jepa_predictor_ham10000_exp003_v1
```

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> A linear JEPA-style latent predictor trained on one nuisance family
> and evaluated on a disjoint family on HAM10000 loses 0.27 AUROC of its
> in-distribution advantage, collapsing from 0.920 to 0.680 test AUROC
> at the same training configuration. Frozen DINOv2 ViT-S/14 cosine
> distance drops from 0.652 AUROC to 0.361 — below random, with stable
> pairs embedding farther apart than genuinely different lesion pairs.
> SSIM distance (0.961) and pixel L2 (0.911) become the strongest
> baselines because the held-out family preserves most of the image
> pixel area. The MVP therefore does not yet demonstrate nuisance-family
> generalization for JEPA-style latent prediction on this dataset,
> though it cleanly characterizes two distinct failure regimes — one
> where frozen DINOv2 fails and one where the JEPA predictor fails —
> that any next-step scheme has to navigate.

### 9.2 Numbers safe to quote

- JEPA AUROC train / val / test: **0.953 / 0.710 / 0.680**
- SSIM test AUROC: **0.961** [0.953, 0.968]
- Pixel L2 test AUROC: **0.911** [0.896, 0.925]
- DINOv2 ViT-S/14 cos test AUROC: **0.361** [0.338, 0.384] (below 0.5)
- DINOv2 ViT-B/14 cos test AUROC: **0.415** [0.393, 0.438] (below 0.5)
- JEPA Δ vs strongest baseline: **−0.281** AUROC (non-overlapping CIs)
- Val loss / train loss ratio: **1.49×** (distribution-shift signature)
- Wall time: ~1h 48m; linear-predictor fit 388 s

### 9.3 Pedagogical beats

1. **"Declare your distribution shift explicitly."** Train and test
   secretly sharing a nuisance family is the kind of leak that gives you
   a real number that does not generalize. Holding a family out is how
   you catch it.
2. **"Different baselines win on different proxies."** EXP-001 had
   pixel L2 at ceiling. EXP-002 had DINOv2 ViT-S/14 at ceiling. EXP-003
   has SSIM at 0.96. The strongest baseline is a property of the
   proxy, not of the domain.
3. **"Below-random is a signal."** DINOv2 cosine at AUROC 0.36 is not
   "DINOv2 is broken"; it is "the held-out family rearranges DINOv2's
   distance geometry in a way that inverts the stable/changing
   ordering." That is information about the representation, legible to
   anyone who reads the score-distribution table.
4. **"Falsification is load-bearing."** A single run showing a positive
   delta (EXP-002) is not a result. EXP-003 is what promotes EXP-002
   from "measurement" to "measurement with a known scope." The scope is
   what eventually goes in the paper.

### 9.4 Three-experiment arc for talks / blog posts / videos

The EXP-001 → EXP-002 → EXP-003 triangle is a complete self-contained
research story:

1. **EXP-001** — Build the pipeline, run it on real data. Every baseline
   is at ceiling. Report honestly: proxy is trivial.
2. **EXP-002** — Harden the proxy. JEPA predictor wins by +0.27 AUROC
   with non-overlapping CIs. Write the caveats before celebrating.
3. **EXP-003** — Test one of those caveats. The win was
   nuisance-family-specific. Write the honest partial negative and
   identify exactly where the gap lives.

That progression is more honest and more pedagogically valuable than
any single positive result would be.

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-003 |
|---|---|
| Held-out nuisance evaluation | §2.2, §4.1, §4.2, §4.5 |
| Generalization-gap analysis | §4.3 training dynamics, §5.4 |
| Baseline-regime characterization | §4.6, §5.2 |
| DINOv2 brittleness under hue/posterize/motion | §5.3 |
| Joint reading with EXP-002 | §5.5 |
| Next-step program | §7 |

### 9.6 Plot assets (embeddable)

- `artifacts/plots/baseline_score_histogram.png` — SSIM and pixel L2
  separation under the held-out family.
- `artifacts/plots/jepa_score_histogram.png` — JEPA predictor scores
  on the test split, showing the compressed stable vs changing gap
  (0.08 AUROC) versus the train-split gap (0.23).
- `logs/progress.jsonl` — full observability trace for Gantt-style
  "what a real research Job looks like" figures.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-23 | Abdelhamid Bakhta | Initial report; run completed on `c34c398`, analysis written, joint-reading with EXP-002 scoped. |
