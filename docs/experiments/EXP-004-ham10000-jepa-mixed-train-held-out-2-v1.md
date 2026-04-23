# EXP-004 — HAM10000 mixed-train / third-family eval (`ham10000-hf-dinov2-exp004-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Second falsification positive. Mixed-family training does not rescue generalization; JEPA collapses to **0.249 AUROC (below-random, inverted)** on a third unseen family, DINOv2 cosine and SSIM also invert, and no dense score separates stable from changing meaningfully. The linear-over-frozen-DINOv2 scaffold is now the clear bottleneck.
**Date (UTC):** 2026-04-23
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dinov2-exp004-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dinov2-exp004-v1/`
**Launcher commit:** `3b1e6ec`

---

## 1. Summary

EXP-003 showed that a JEPA predictor trained on one nuisance family does
not transfer to a second disjoint family. EXP-004 asked whether training
on **two** disjoint families at once closes that gap by testing
generalization to a **third** unseen family.

The answer is no — and the failure mode is instructive. Key numbers on
the test split (N = 2,000 pairs, held-out stable family =
`strong_held_out_2`):

| Score | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| **JEPA predictor (exp004)** | **0.249** | [0.230, 0.270] | **stable > changing (inverted)** |
| SSIM distance | 0.436 | [0.411, 0.460] | stable > changing (inverted) |
| DINOv2 ViT-S/14 cosine | 0.306 | [0.286, 0.326] | stable > changing (inverted) |
| DINOv2 ViT-B/14 cosine | 0.274 | [0.253, 0.294] | stable > changing (inverted) |
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct) |

Pixel L2 is now the only dense score with the correct stable-vs-changing
sign, and its margin above chance is small (0.58). Every other score
gives stable pairs *larger* distances than changing pairs — the third
family shifts all representation-level distances more than a same-
diagnosis-same-site lesion swap does.

The JEPA predictor learned its training distribution well — **train
AUROC = 0.900**, almost matching EXP-003's 0.953 despite having to fit
two families with the same linear capacity — and then actively hurt
generalization by pushing outputs along a mixture of `strong` and
`strong_held_out` directions, which is close to *changing*-pair
embeddings in DINOv2 space and far from `strong_held_out_2` stable
targets. The predictor is now worse than a coin flip, in the same
direction as frozen DINOv2.

Collapse checks still pass. Leakage probes still zero. Training loss
dropped cleanly; val loss is **2× train loss**, the strongest
distribution-shift signature of the four runs so far.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-003

One dataset-level change. Everything else — same seed, same HAM10000
mount, same DINOv2 backbones, same strict changing-pair policy, same
linear predictor.

| Knob | EXP-003 | EXP-004 |
|---|---|---|
| `dataset.nuisance_severity` (train stable pairs) | `strong` | **`strong,strong_held_out`** (deterministic rotation by pair index) |
| `dataset.nuisance_severity_eval` (val + test stable pairs) | `strong_held_out` | **`strong_held_out_2`** |

### 2.2 Three disjoint families, no shared transform

| Family | Where used | Transforms |
|---|---|---|
| `strong` | train (even pair indices) | brightness, contrast, saturation, rotation, hflip, scale + translate, Gaussian blur, Gaussian noise, JPEG 45–70 |
| `strong_held_out` | train (odd pair indices) | hue shift, posterize, sharpen (unsharp-mask), motion blur, random rectangular erasing, JPEG 20–40 |
| `strong_held_out_2` | val + test | gamma correction, colour-temperature shift, radial vignette, salt-and-pepper impulse noise, JPEG 80–95 |

Unit test pins that the three families have no shared recipe keys —
they are genuinely orthogonal in transform type and hyperparameter
range.

### 2.3 Held constant

- Dataset: HAM10000 via `abdelstark/ham10000`, 10,015 images.
- Seed: 20260422.
- Splits: 5,229 / 1,120 / 1,121 lesions, zero lesion-ID overlap.
- Pair counts: 1,000 stable + 1,000 changing per split, strict
  `same_diagnosis_and_site` changing-pair policy.
- Embedding backbones: DINOv2 ViT-S/14 + ViT-B/14, frozen.
- JEPA predictor: identity-initialized linear map on DINOv2 ViT-B/14,
  200 epochs, batch 128, LR 0.03, weight decay 0.001.
- Metric protocol: 1,000-sample bootstrap CI, 95 % level, fixed TPR 0.80.

---

## 3. Operational timeline

`logs/progress.jsonl` captured the full run.

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Same pinned wheel resolve as EXP-002/003 |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~25 min | Train: 500 × `strong` + 500 × `strong_held_out` per split; val + test: 1,000 × `strong_held_out_2` per split |
| DINOv2 ViT-S/14 + ViT-B/14 embedding export | ~30 min | Same 8,004 unique images |
| Baselines (test split) | ~5 min | pixel L2, SSIM, DINOv2 cosines |
| JEPA linear predictor fit | **22 s** | Shortest of all four runs — same numerical problem, scheduler noise |
| Upload | ~3.5 min | 3,014 files, ~279 MB |
| **Total wall time** | **≈ 2h** | Similar envelope to EXP-001/002/003 |

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct) |
| SSIM distance | 0.436 | [0.411, 0.460] | stable > changing (inverted) |
| DINOv2 ViT-S/14 cosine | 0.306 | [0.286, 0.326] | stable > changing (inverted) |
| DINOv2 ViT-B/14 cosine | 0.274 | [0.253, 0.294] | stable > changing (inverted) |
| **JEPA predictor (exp004)** | **0.249** | [0.230, 0.270] | **stable > changing (inverted)** |

Non-overlap check: JEPA's upper bound (0.270) is below every other
score's lower bound. The predictor is decisively worse than every
cheap baseline, worse than chance (0.5), and worse than both DINOv2
cosines it is built on top of.

### 4.2 JEPA across splits

| Split | Family used for stable target | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` (mixed) | **0.900** | [0.886, 0.912] |
| val | `strong_held_out_2` | 0.265 | [0.244, 0.287] |
| test | `strong_held_out_2` | 0.249 | [0.230, 0.270] |

Train AUROC fell modestly from EXP-003's 0.953 to 0.900 — expected,
because the linear predictor now has to approximate two disjoint vector
fields with the same capacity. Generalization to the third family
dropped by **0.65 AUROC** from train to test.

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000787 | 0.001188 |
| 20 | 0.000691 | 0.001435 |
| 100 | 0.000689 | 0.001433 |
| 200 | 0.000688 | 0.001429 |

Val loss is **≈ 2.07 × train loss** (0.00143 / 0.00069), the starkest
distribution-shift signature of any run in this sequence:

| Run | Train loss (final) | Val loss (final) | Ratio |
|---|---:|---:|---:|
| EXP-002 (matched) | 0.000082 | 0.000075 | 0.91 (val < train, small gap) |
| EXP-003 (one held-out) | 0.000562 | 0.000837 | 1.49 |
| **EXP-004 (third held-out)** | **0.000688** | **0.001429** | **2.07** |

Val loss actually *increases* slightly between epoch 1 and epoch 20,
then stabilizes. The predictor is fitting the training mixture in a way
that becomes progressively *less* helpful for the held-out third
family before plateauing.

### 4.4 Representation health

- `prediction_norm_mean` = 1.0
- `prediction_norm_min` = 0.9999
- `dimension_variance_mean` ≈ 4.7 × 10⁻⁴ (consistent with EXP-002/003)
- `collapsed` = **False**

The predictor is not collapsed; it is confidently wrong.

### 4.5 JEPA pair-score distributions across splits

| Split | Stable mean | Changing mean | Gap (stable − changing) |
|---|---:|---:|---:|
| train | 0.268 | 0.461 | **−0.194 (correct)** |
| val | 0.538 | 0.424 | **+0.114 (inverted)** |
| test | 0.537 | 0.417 | **+0.120 (inverted)** |

On train, the predictor scores stable pairs meaningfully smaller than
changing pairs — the intended ordering. On val and test, the ordering
flips by ~0.11 drift units. The predictor's output of "predicted
`strong`-or-`strong_held_out` variant of source A" lies closer to a
different lesion's embedding than to the `strong_held_out_2` variant
of A.

### 4.6 Baseline score distributions (test split)

| Baseline | Stable mean | Changing mean | Direction |
|---|---:|---:|---|
| Pixel L2 | 0.157 | 0.174 | changing > stable (correct, weak) |
| SSIM distance | 0.469 | 0.450 | stable > changing (inverted) |
| DINOv2 ViT-S/14 cosine | 0.383 | 0.303 | stable > changing (inverted) |
| DINOv2 ViT-B/14 cosine | 0.425 | 0.324 | stable > changing (inverted) |

Pixel L2 remains correctly oriented because salt-and-pepper noise +
small gamma + mild vignette change pixels only modestly, so
`(source, source + small perturbation)` is still closer pixel-wise than
`(lesion A, lesion B)`. But SSIM and both DINOv2 cosines now think
stable is farther than changing because those measures react
non-linearly to global tonal reshaping — gamma and colour temperature
in particular move DINOv2 features more than a different-lesion swap
does.

---

## 5. Analysis

### 5.1 Mixed-family training did not transfer

Direct test of the EXP-004 hypothesis:

> If the JEPA predictor sees two disjoint nuisance families at train
> time, does it generalize to a third unseen family?

Result: **no**. JEPA test AUROC = 0.249 on `strong_held_out_2`. Even
setting aside the inversion, **0.5 − 0.25 = 0.25 AUROC worse than
random**. The hypothesis is rejected at this scale with this scaffold.

### 5.2 Why the predictor inverted

The JEPA objective maps `v(source) → E_f[v(f(source))]` where `f` is
sampled from the training families. After EXP-004's mixed training, the
predictor's output for a test image A is roughly "the average DINOv2
embedding of a `strong` variant and a `strong_held_out` variant of A".

At test time:

- A **stable** pair is `(A, f_3(A))` where `f_3` ∈ `strong_held_out_2`.
  The predictor outputs something in the `{strong, strong_held_out}`
  mixture manifold, while `f_3(A)` lives in an orthogonal region of
  DINOv2 space that the predictor never learned to reach. So the
  predicted→actual distance is large.
- A **changing** pair is `(A, B)` where B is a different lesion of the
  same diagnosis and site. `v(B)` is a "clean" DINOv2 embedding of a
  real lesion image — exactly the region where the
  `{strong, strong_held_out}` mixture manifold *approximately* lives
  (because variants of lesion images are close to other lesion images
  on DINOv2's lesion manifold). So the predicted→actual distance is
  *smaller* for changing pairs than for held-out-family stable pairs.

The net result is the inversion observed in §4.5: stable scores are
higher than changing scores. Every other DINOv2-based score has the
same inversion for the same structural reason (gamma + colour
temperature + vignette shift DINOv2 embeddings more than a
same-dx-site lesion swap does).

### 5.3 DINOv2 is the root-cause bottleneck, not JEPA

The cleanest way to read EXP-004 is as a DINOv2 diagnostic:

- DINOv2 cosine AUROC on the three eval families:
  - EXP-002 (matched `strong`): 0.65 (correct direction)
  - EXP-003 (`strong_held_out`): 0.36 (inverted)
  - EXP-004 (`strong_held_out_2`): 0.27 (inverted, further from 0.5)

- JEPA AUROC on the three eval families, same codebase:
  - EXP-002 (matched `strong`): 0.92 (correct direction)
  - EXP-003 (`strong_held_out`): 0.68 (still correct, but weak)
  - EXP-004 (`strong_held_out_2`): 0.25 (inverted)

Wherever frozen DINOv2 cosine goes, the linear predictor goes with it.
Mixed-family training gave the predictor more robustness at train time
(it had to fit two fields) and it did (train AUROC = 0.90), but it did
not teach the predictor to reach into DINOv2-space regions neither
training family touches. The linear map is not the right class of
function to bridge that gap.

### 5.4 The joint reading across four runs

The four experiments now have a consistent picture:

| Run | Proxy | Train → Test AUROC | JEPA win over strongest baseline |
|---|---|---:|:---|
| EXP-001 | trivial | ≈ 1.0 → ≈ 1.0 | not interpretable (ceiling) |
| EXP-002 | hardened, matched eval | 0.95 → 0.92 | **+0.27 AUROC, non-overlapping CIs** |
| EXP-003 | hardened, one-family held-out eval | 0.95 → 0.68 | **−0.28** (loses to SSIM) |
| EXP-004 | hardened, mixed train, third-family held-out eval | 0.90 → 0.25 | **−0.33** (loses to pixel L2; below random) |

Every generalization step worsens the gap. The narrow win in EXP-002 is
family-specific, the partial loss in EXP-003 is distribution-shift on
top of a DINOv2 whose distance geometry inverts, and the severe loss in
EXP-004 is "the distance geometry inverted even further and the
predictor amplified the inversion."

### 5.5 What the thesis should say now

Honest spec-compliant statement of the MVP claim, after four runs:

> On a leakage-controlled HAM10000 longitudinal-proxy task, a linear
> JEPA-style predictor over frozen DINOv2 ViT-B/14 beats cheap
> baselines (pixel L2, SSIM, DINOv2 cosine) by a large margin when the
> nuisance distribution at test time matches training. It loses on an
> unseen nuisance family, and it loses even more severely on a third
> unseen family after mixed-family training. The scaffold as currently
> instantiated is a family-specific corrector rather than a
> nuisance-invariant lesion representation, and the binding constraint
> is the linear-over-frozen-DINOv2 choice, not the JEPA objective
> itself. Any claim of nuisance invariance from this stack requires
> either a more expressive predictor class, a trainable backbone, or a
> richer mixture of training nuisance — and each has been scoped as a
> distinct follow-up experiment.

---

## 6. Limitations and threats to validity

1. **Linear predictor.** EXP-004 is the cleanest evidence that the
   linear scaffold cannot extrapolate to unseen nuisance geometries.
   A deeper predictor might generalize; it might also overfit further.
2. **Frozen DINOv2.** All four runs share one backbone. A different
   frozen backbone (OpenCLIP, DreamSim, or a dermatology-tuned encoder)
   could have different distance-geometry behaviour under these
   families.
3. **Single seed per run.** Seed variance has not yet been measured;
   the effects here are large enough that seed luck is unlikely to flip
   the direction, but the confidence intervals are bootstrap CIs, not
   seed CIs.
4. **Nuisance families are synthetic.** All three families are
   procedural; real smartphone re-photography imposes a different,
   unknown distribution. The MVP is still cross-sectional.
5. **`strong_held_out_2` is itself a design choice.** A different third
   family (e.g. elastic deformation + optical flow warp) would produce
   different numbers. The *direction* of the generalization story is
   robust across EXP-003 and EXP-004, but the magnitudes are
   family-specific.

---

## 7. What changes for the next run (EXP-005 scoping)

The bottleneck is now clearly identified. The next experiment should
attack it head-on.

Priority list, honest:

1. **Small non-linear predictor.** 2-layer MLP, hidden dim 512, residual
   to identity, same training budget. If this generalizes to
   `strong_held_out_2` where the linear map failed, the scaffold was
   the bottleneck. If not, the backbone is.
2. **Backbone swap.** Re-run EXP-002 / EXP-003 / EXP-004 configs with
   OpenCLIP ViT-B/16 or DreamSim as the frozen encoder. Quantify how
   much of the failure mode is DINOv2-specific.
3. **Seed sweep.** Three seeds × {EXP-002, EXP-003, EXP-004}.
4. **Richer mixed training.** Train on all three families
   simultaneously, evaluate on a fourth held-out family; separates
   "training diversity" from "predictor capacity."
5. **Per-diagnosis / per-site breakdown.** For EXP-002 especially, does
   the JEPA win hold across rare diagnoses?

Explicitly not yet:

- DINOv2 fine-tuning — still premature given the predictor-class
  question is unresolved.
- Abandoning the project. EXP-001–004 is a publishable arc that
  characterizes a specific failure surface. Shipping it honestly is a
  better outcome than trying to force a win.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-exp004-v1 \
  ./scripts/hf_jobs_ham10000_exp004.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dinov2-exp004-v1
```

Expected top line: `auroc: 0.2491`, `strongest_baseline: pixel_l2 =
0.5802`, `delta_vs_baseline: -0.3311`, `collapsed: False`,
`tier: public`.

### 8.3 Config diff vs EXP-003

```diff
 dataset:
-  nuisance_severity: strong
-  nuisance_severity_eval: strong_held_out
+  nuisance_severity: "strong,strong_held_out"
+  nuisance_severity_eval: strong_held_out_2
 training:
-  model_id: jepa_predictor_ham10000_exp003_v1
+  model_id: jepa_predictor_ham10000_exp004_v1
```

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Training a linear JEPA-style latent predictor over frozen DINOv2
> ViT-B/14 on *two* disjoint HAM10000 nuisance families and evaluating
> on a *third* unseen family produces a below-random test AUROC of
> 0.249 — the predictor assigns higher drift scores to stable pairs
> than to genuinely different lesions. SSIM distance (AUROC 0.436) and
> both DINOv2 cosines (0.306, 0.274) also invert under the third
> family. Only pixel L2 (AUROC 0.580) keeps the correct stable-vs-
> changing sign. Mixed-family training does not repair the
> generalization gap from EXP-003; the linear-over-frozen-DINOv2
> scaffold is the limiting factor, not the diversity of the training
> nuisance distribution.

### 9.2 Numbers safe to quote

- JEPA AUROC train / val / test: **0.900 / 0.265 / 0.249**
- Pixel L2 test AUROC: 0.580 [0.556, 0.606] (only correctly-signed score)
- SSIM test AUROC: 0.436 [0.411, 0.460] (inverted)
- DINOv2 ViT-S/14 cosine test AUROC: 0.306 [0.286, 0.326] (inverted)
- DINOv2 ViT-B/14 cosine test AUROC: 0.274 [0.253, 0.294] (inverted)
- JEPA Δ vs strongest (pixel L2): **−0.331**, non-overlapping CIs
- Val / train loss ratio: **2.07** (highest in the series)
- JEPA stable mean − changing mean: **+0.12** (inverted from correct −0.19 on train)

### 9.3 Pedagogical beats

1. **"Below-random is worse than random."** A score at 0.25 AUROC is
   not 'not working'; it is working *in the exact wrong direction*. A
   simple sign flip would give 0.75 AUROC. That fact, alone, tells you
   the predictor has learned a real structure — it is just aligned
   against the task rather than with it.
2. **"Training diversity is not a free fix."** EXP-004 tried to solve
   EXP-003 by showing the model more families at train time. It
   instead exposed a capacity limit of the predictor class.
3. **"Your backbone is always in scope."** EXP-002 through EXP-004 all
   used the same frozen DINOv2; the generalization collapse is as much
   a story about DINOv2's distance-geometry brittleness as it is about
   JEPA's predictor class.
4. **"Write the failure surface, not the headline."** A 0.25 AUROC is
   a paper-worthy result if it is characterized. The three inversion
   directions across EXP-003 and EXP-004, the consistent
   train→val→test drop per run, and the cross-run comparison table are
   the publishable object.

### 9.4 Cross-run table (the single most useful asset)

| Run | Proxy config | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---:|---|---:|
| EXP-001 | trivial | 0.9985 / 0.9996 / 0.9998 | DINOv2 ViT-S/14 cos = 1.0000 | −0.0001 |
| EXP-002 | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2 ViT-S/14 cos = 0.652 | **+0.269** |
| EXP-003 | hardened, one-family held-out eval | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| **EXP-004** | hardened, mixed train + third-family eval | **0.900 / 0.265 / 0.249** | Pixel L2 = 0.580 | **−0.331** |

### 9.5 Plot assets (embeddable)

- `artifacts/plots/baseline_score_histogram.png` — four baselines under
  `strong_held_out_2`, showing the three inverted score orderings.
- `artifacts/plots/jepa_score_histogram.png` — JEPA test-split drift
  scores, showing the inverted stable / changing separation.
- `logs/progress.jsonl` — per-stage timings for any Gantt-style figure.

### 9.6 Paper-section mapping

| Paper section | Drawn from EXP-004 |
|---|---|
| Mixed-family training ablation | §2.1, §2.2, §4.2 |
| Below-random analysis and inversion | §4.1, §4.5, §4.6, §5.2 |
| DINOv2 distance-geometry characterization | §5.3 |
| Bottleneck identification (predictor class) | §5.3, §7 |
| Four-run honest joint reading | §5.4, §5.5 |
| Next-step program | §7 |

### 9.7 Four-experiment arc for talks / articles / videos

EXP-001 → EXP-002 → EXP-003 → EXP-004 is now a clean four-act structure.
Each act produced *evidence* rather than a *desired outcome*:

1. Build the whole pipeline. Proxy trivially separable. Report honestly.
2. Harden the proxy. JEPA wins +0.27 AUROC with non-overlapping CIs. Write the caveats.
3. Hold out a nuisance family at eval. Win collapses. SSIM becomes king.
4. Train on two families, evaluate on a third. Predictor inverts. Below-random. Bottleneck identified: the linear-over-frozen-DINOv2 scaffold.

This arc is more convincing than a cherry-picked positive because it
demonstrates *falsification working as intended*, and it ends with a
concretely scoped next experiment rather than a vague gesture.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-23 | AbdelStark + Claude Code | Initial report; run completed on `3b1e6ec`, bottleneck identified as linear-over-frozen-DINOv2, EXP-005 scope locked. |
