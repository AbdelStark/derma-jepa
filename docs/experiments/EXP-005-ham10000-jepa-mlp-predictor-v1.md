# EXP-005 — HAM10000 MLP predictor on the EXP-004 proxy (`ham10000-hf-dinov2-exp005-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Third consecutive negative on the held-out direction, with an unexpected twist: the MLP barely departed from identity under the EXP-002 hyperparameters, so the "scaffold vs backbone" falsification is *partially* completed. The backbone is implicated; the MLP-capacity question needs EXP-006 to re-run with a properly tuned optimiser before it is fully settled.
**Date (UTC):** 2026-04-23 → 2026-04-24
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dinov2-exp005-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dinov2-exp005-v1/`
**Launcher commit:** `47cbafe`

---

## 1. Summary

EXP-004 identified the linear-over-frozen-DINOv2 scaffold as a candidate
bottleneck: the linear predictor fit two disjoint training nuisance
families well (train AUROC 0.900) and then inverted below random on a
third unseen family (test AUROC 0.249). EXP-005 held everything else
fixed and only swapped the predictor class: a 2-layer MLP with identity
residual `y = x + W2 ReLU(W1 x + b1) + b2`, hidden dim 512, identical
training budget and optimiser.

Test AUROC is essentially unchanged: **0.270** [0.249, 0.293]. Pixel L2
is still the strongest baseline at 0.580. The JEPA delta vs the
strongest baseline is **−0.310**, statistically indistinguishable from
EXP-004's −0.331 and directionally identical.

However, the training dynamics are unexpected in an informative way:

- **Train AUROC: 0.572** — far lower than EXP-004 linear (0.900).
- **Train loss barely moved**: 0.000925 → 0.000872 over 200 epochs (~6%
  reduction), vs EXP-004 linear's 13% reduction from a lower starting
  point.
- **Val loss *increased* monotonically** from 0.001108 to 0.001120
  while train loss decreased.
- **MLP predictor scores track the frozen DINOv2 ViT-B/14 cosine
  baseline almost exactly** (test AUROC 0.270 vs 0.274).

Interpretation: under the EXP-002 optimizer configuration (SGD, LR =
0.03, weight decay = 0.001, W2 initialised at zero for the identity
warm-start), the MLP stayed close to identity and never meaningfully
departed from passing DINOv2 embeddings through unchanged. So the
measured AUROC reflects what DINOv2 alone does on `strong_held_out_2`,
not what an expressive predictor does.

This leaves the §7 scope for EXP-006 more sharply defined, not less:
the backbone-bottleneck hypothesis is *consistent* with EXP-005 but
*not* cleanly confirmed, because the MLP never actually fit the
training distribution to test whether capacity would transfer. EXP-006
needs to do two things: (1) retrain the MLP with an optimiser that
escapes the zero-W2 attractor (Adam, higher LR, or no weight decay on
residual layers) so we can see what a properly-fit MLP does on
`strong_held_out_2`; and (2) the backbone swap (OpenCLIP or DreamSim)
that is now clearly motivated regardless of (1)'s outcome.

Collapse checks still pass. Lesion-ID leakage probes still return zero.

---

## 2. Experimental setup

### 2.1 What changed vs EXP-004

One knob only. Predictor class. Everything else is byte-identical to
EXP-004 so the delta is a clean read.

| Knob | EXP-004 | EXP-005 |
|---|---|---|
| `training.predictor` | `linear` (identity + small noise) | **`mlp`** (2-layer, identity residual) |
| `training.hidden_dim` | — | **512** |
| Training epochs / batch / LR / weight decay | 200 / 128 / 0.03 / 0.001 | 200 / 128 / 0.03 / 0.001 |

The MLP is:

```
y = x + W2 · ReLU(W1 · x + b1) + b2
```

with `W1` (768 × 512) initialised Kaiming uniform × 0.1, `b1 = 0`,
`W2` (512 × 768) **zero-initialised**, `b2 = 0`. This makes the
function at step 0 an exact identity, matching the linear predictor's
identity warm-start so the two classes compete under identical starting
conditions. Optimiser: torch SGD with the same learning rate and weight
decay as the linear runs.

### 2.2 Held constant

Same as EXP-004: dataset `abdelstark/ham10000`, seed 20260422, lesion-
ID splits (5,229 / 1,120 / 1,121), 1,000 stable + 1,000 changing pairs
per split, strict same-diagnosis-site changing-pair policy, DINOv2
ViT-S/14 + ViT-B/14 backbones, bootstrap CI protocol. Train stable
pairs rotate between `strong` and `strong_held_out`; val and test use
`strong_held_out_2`.

---

## 3. Operational timeline

Full observability stream in `logs/progress.jsonl`. Timings follow the
EXP-004 envelope because only the linear-predictor fit step changes,
and that step is now GPU-backed instead of numpy:

| Stage | Wall time | Notes |
|---|---:|---|
| Scheduling + install | ~5 min | Pinned wheel + `[model]` extras |
| Manifest build (incl. 3,000 stable-variant PNGs across 3 families) | ~25 min | Same distribution as EXP-004 |
| DINOv2 S/14 + B/14 embedding export | ~30 min | Identical image set |
| Baselines (test) | ~5 min | Same four cheap baselines |
| JEPA MLP predictor fit | **1.4 s** | torch + A10G, 200 epochs × 1,000 pairs × (768 → 512 → 768). Down from EXP-004's 22 s numpy run; scheduler noise is now the dominant variance |
| Upload | ~3.5 min | 3,014 files, ~280 MB |
| **Total wall time** | **≈ 2 h** | Essentially unchanged from EXP-004 |

The predictor-fit time is negligible. The Job's cost is dominated by
FUSE-backed image reads during embedding export, which is invariant to
predictor class.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | Direction |
|---|---:|:---:|---|
| Pixel L2 | **0.580** | [0.556, 0.606] | changing > stable (correct) |
| SSIM distance | 0.436 | [0.411, 0.460] | stable > changing (inverted) |
| DINOv2 ViT-S/14 cosine | 0.306 | [0.286, 0.326] | stable > changing (inverted) |
| **DINOv2 ViT-B/14 cosine** | **0.274** | [0.253, 0.294] | stable > changing (inverted) |
| **JEPA predictor (exp005, MLP)** | **0.270** | [0.249, 0.293] | **stable > changing (inverted)** |

The MLP predictor's AUROC is **within one bootstrap CI width of the
DINOv2 ViT-B/14 cosine baseline** (0.270 vs 0.274). Operationally the
MLP is behaving as a near-identity pass-through of DINOv2 embeddings.

### 4.2 JEPA across splits (the crucial diagnostic)

| Split | Stable family | AUROC | 95% CI |
|---|---|---:|:---:|
| train | `strong` + `strong_held_out` | **0.572** | [0.546, 0.595] |
| val | `strong_held_out_2` | 0.293 | [0.270, 0.315] |
| test | `strong_held_out_2` | 0.270 | [0.249, 0.293] |

The train AUROC is the surprise. EXP-004 linear hit 0.900 on the same
training distribution. EXP-005 MLP hit 0.572 — barely above random.
The MLP did not even fit the training distribution well, which means
EXP-005 cannot cleanly test "does a well-fit MLP generalise" — it only
tests "does a barely-trained MLP generalise."

### 4.3 Training dynamics

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000925 | 0.001108 |
| 20 | 0.000920 | 0.001109 |
| 100 | 0.000897 | 0.001113 |
| 200 | 0.000872 | 0.001120 |

Train loss drops 5.7 % in total. Val loss **increases monotonically**
from 0.001108 to 0.001120 — the predictor is moving slightly in a
direction that helps train and hurts val. This is a classic "barely
fitting, slight overfit to tiny movement" pattern, not the "fits train
tightly, fails to transfer" pattern EXP-004 showed.

Cross-run comparison of loss trajectories:

| Run | Predictor | Final train loss | Final val loss | Train loss delta |
|---|---|---:|---:|:---:|
| EXP-002 | linear | 0.000082 | 0.000075 | −10 % |
| EXP-003 | linear (held-out eval) | 0.000562 | 0.000837 | −13 % |
| EXP-004 | linear (mixed + third held-out) | 0.000688 | 0.001429 | −13 % |
| **EXP-005** | **MLP (mixed + third held-out)** | **0.000872** | **0.001120** | **−6 %** |

EXP-005's train-loss delta is half of every prior run. The MLP is
underfitting.

### 4.4 Why the MLP underfit

Under the EXP-002/003/004 optimizer configuration (SGD, LR 0.03,
weight decay 0.001), applied to an MLP whose W2 is initialised at zero:

- At step 0 the residual `W2 ReLU(W1 x + b1) + b2` is exactly zero, so
  `y = x` and the only gradient signal comes from the MSE between the
  identity output and the heterogeneous training target distribution.
- SGD's weight_decay term pulls `W1, W2, b1, b2` toward zero, which for
  `W2` and `b2` is the identity-predictor attractor. The weight decay
  and the "stay at zero residual" init pressure are *the same force*.
- The gradient through `W2` at zero-W2 is driven by `∂loss/∂W2 =
  ReLU(W1 x + b1)^T · err`, which is small when `W1` is Kaiming × 0.1.
  So the escape velocity from the identity attractor is low under this
  optimiser.

Net effect: the MLP stayed close to identity, the residual never grew
large, and the predictor's test-time behaviour fell back to "pass
DINOv2 embeddings through unchanged." Which is exactly why the MLP
AUROC (0.270) matches the DINOv2 ViT-B/14 cosine baseline (0.274).

This is a training-configuration issue, not a capacity issue. The
linear predictor's `weight - identity` regulariser pulled it toward a
*non-zero-residual* fixed point (the identity matrix); the MLP's zero
fixed point pulls it toward *zero residual* (pass-through). The two
are not analogous regularisations despite being called "weight decay"
in both runs.

### 4.5 Representation health

- `prediction_norm_mean` = 1.000
- `prediction_norm_min` = 0.9999999
- `dimension_variance_mean` = 5.0 × 10⁻⁴
- `dimension_variance_min` = 2.3 × 10⁻⁶
- `collapsed` = **False**

Unchanged from the three prior runs. No representational collapse.
Dimension variance min dropped roughly 10× from EXP-002 (1.7 × 10⁻⁵ →
2.3 × 10⁻⁶), a faint signal that some embedding dimensions have
lower variance in the predictor output, but still above the collapse
threshold.

### 4.6 JEPA pair-score distributions across splits

| Split | Stable mean | Changing mean | Gap (stable − changing) |
|---|---:|---:|---:|
| train | 0.339 | 0.371 | −0.032 (correct, weak) |
| val | 0.433 | 0.338 | **+0.095 (inverted)** |
| test | 0.432 | 0.328 | **+0.104 (inverted)** |

Train gap shrinks to a third of EXP-004's, consistent with underfitting.
Val + test gaps keep the inverted orientation DINOv2 imposes on this
family — confirming that the MLP, unable to move the predictor far from
identity, inherits DINOv2's distance geometry unchanged.

---

## 5. Analysis

### 5.1 What EXP-005 actually proved

**Proved:** Under identical optimizer and regulariser to EXP-002/003/
004, a 2-layer MLP with an identity-residual initialisation does not
escape the near-identity attractor within 200 epochs at LR 0.03. The
resulting predictor falls back to the DINOv2 ViT-B/14 cosine baseline
in behaviour, which is below-random on `strong_held_out_2`.

**Not yet proved:** That a *well-fit* MLP on the same training
distribution would fail to generalise. Since the MLP underfit, the
"scaffold vs backbone" bottleneck test is inconclusive in the
capacity direction. The evidence that remains for "backbone is the
bottleneck" is that every predictor variant tried so far — linear,
near-identity MLP, and by extension frozen DINOv2 cosine alone — all
land at essentially the same test AUROC (≈ 0.25–0.27) on this family,
which is consistent with the backbone imposing the floor.

### 5.2 The five-run picture so far

| Run | Predictor | Proxy | Train → Test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---|---|---:|
| EXP-001 | linear | trivial | 1.00 → 1.00 | DINOv2-S ≈ 1.00 | ≈ 0 |
| EXP-002 | linear | hardened, matched eval | 0.95 → 0.92 | DINOv2-S = 0.65 | **+0.27** |
| EXP-003 | linear | hardened, one-family held-out eval | 0.95 → 0.68 | SSIM = 0.96 | −0.28 |
| EXP-004 | linear | hardened, mixed train + third-family held-out | 0.90 → 0.25 | Pixel L2 = 0.58 | −0.33 |
| **EXP-005** | **MLP** | **same as EXP-004** | **0.57 → 0.27** | Pixel L2 = 0.58 | **−0.31** |

Reading the column `Train → Test AUROC` across the last three runs:

- EXP-003: 0.95 → 0.68. Linear fit training tightly, lost ~0.27 in
  generalisation.
- EXP-004: 0.90 → 0.25. Linear fit training well even on two families,
  lost ~0.65 in generalisation (inverted).
- EXP-005: 0.57 → 0.27. MLP barely fit training, landed at the DINOv2
  cosine baseline for test.

The test AUROCs in EXP-004 (0.25) and EXP-005 (0.27) are
indistinguishable within a single bootstrap CI, despite the predictor
classes being different. That *is* evidence for the backbone
bottleneck — whatever predictor you use, the test AUROC floor is set
by what DINOv2 does, not by how expressive the function on top is —
but the evidence would be cleaner if EXP-005's MLP had actually fit
the training distribution.

### 5.3 What a properly-fit MLP would likely show

Three possible outcomes for a better-tuned MLP (EXP-006):

1. Train AUROC ≈ 0.95, test AUROC ≈ 0.25 (still inverted). The
   scaffold was the bottleneck for training fit but not for
   generalisation; the backbone is the root cause.
2. Train AUROC ≈ 0.95, test AUROC ≈ 0.5 (flat, non-inverted). Improved
   expressivity closes part of the gap; more training diversity would
   likely help further.
3. Train AUROC ≈ 0.95, test AUROC ≈ 0.75+. The scaffold was the full
   bottleneck; the thesis is alive at this scale.

I expect (1) on the strength of the EXP-004 evidence: the linear
predictor in EXP-004 DID fit training to 0.90 and still collapsed to
0.25 on test. A more expressive MLP would likely fit training even
tighter and still face the same backbone-induced inversion at test
time. But (2) and (3) are not ruled out until EXP-006 runs.

### 5.4 Thesis-level statement after five runs

> On a leakage-controlled HAM10000 longitudinal-proxy task, a linear
> JEPA-style predictor over frozen DINOv2 ViT-B/14 beats cheap
> baselines by +0.27 AUROC when the test nuisance matches training,
> loses decisively on one held-out family, and loses even more severely
> on a third held-out family after mixed-family training. Swapping the
> predictor class to a 2-layer MLP under the same optimiser does not
> change the test AUROC (0.270 vs 0.249) because the MLP, with
> identity-residual initialisation and weight-decay-toward-zero, stays
> near identity and inherits DINOv2's distance geometry. The
> observation that every predictor variant tried lands at the DINOv2
> cosine baseline on the unseen family is consistent with the backbone
> imposing the performance floor, but the MLP capacity hypothesis has
> not been fully tested because EXP-005's MLP underfit the training
> distribution. EXP-006 addresses both threads: a properly-fit MLP and
> a backbone swap.

---

## 6. Limitations and threats to validity

1. **EXP-005's MLP underfit.** This is the central caveat. The
   optimizer + regulariser combination copied from the linear runs is
   not appropriate for an identity-residual MLP, and the predictor
   never meaningfully departed from its identity initialisation. Any
   claim about MLP capacity from EXP-005 alone is premature.
2. **Still one backbone.** Five runs, same frozen DINOv2 ViT-B/14 in
   every one. Every observation about distance geometry inversion is
   DINOv2-specific.
3. **Still one seed.** Seed sweep still pending (EXP-006/007 scope).
4. **Still one third-family design.** `strong_held_out_2` is one
   choice of "disjoint" — a different disjoint family could give
   different distance-geometry behaviour.
5. **HAM10000 is cross-sectional.** Every longitudinal caveat from
   EXP-001 onward still holds.

---

## 7. What changes for the next run (EXP-006 scoping)

The problem set-up is now split cleanly in two. EXP-006 should do
*both*, either as one run or as two:

1. **Properly tuned MLP predictor.** Same architecture as EXP-005,
   different optimizer. Options, ranked by minimal intervention:
   - Switch SGD → Adam (LR 1e-3, weight decay 1e-4).
   - Remove weight decay on `W2` and `b2` (only regularise `W1`).
   - Initialise `W2` small random instead of zero, accepting that the
     initial function is no longer exact identity.
   - Longer training budget (600 epochs) since the MLP clearly needs
     more steps to escape the attractor.
   I'd run option 1 first — it's the single change that most commonly
   solves zero-init MLP convergence issues.
2. **Backbone swap.** Re-run EXP-002 / EXP-003 / EXP-004 configs under
   OpenCLIP ViT-B/16 (or DreamSim) frozen embeddings. Tells us how
   much of the inversion under `strong_held_out` and
   `strong_held_out_2` is DINOv2-specific. Most informative single
   comparison: EXP-004 config under OpenCLIP vs under DINOv2.

Explicitly still not yet:

- DINOv2 fine-tuning. The predictor-vs-backbone question is not yet
  cleanly separated.
- Abandoning the project. The five-run arc characterises the failure
  surface precisely and is publishable as methodology even in its
  current form.

---

## 8. Reproducibility

### 8.1 Launch command (as run)

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-exp005-v1 \
  ./scripts/hf_jobs_ham10000_exp005.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dinov2-exp005-v1
```

Expected top line: `auroc: 0.2702`, `strongest_baseline: pixel_l2 =
0.5802`, `delta_vs_baseline: -0.3101`, `collapsed: False`,
`tier: public`.

### 8.3 Config diff vs EXP-004

```diff
 training:
-  model_id: jepa_predictor_ham10000_exp004_v1
+  model_id: jepa_predictor_ham10000_exp005_v1
   epochs: 200
   batch_size: 128
   learning_rate: 0.03
   weight_decay: 0.001
+  predictor: mlp
+  hidden_dim: 512
```

---

## 9. Assets for future writeups

### 9.1 Quotable headline

> Replacing the linear JEPA-style predictor with a 2-layer MLP with
> identity-residual initialisation on the same HAM10000 proxy, under
> the same optimiser, leaves test AUROC essentially unchanged at 0.270
> [0.249, 0.293] — statistically indistinguishable from frozen DINOv2
> ViT-B/14 cosine (0.274). The MLP barely moved from its identity
> initialisation (train AUROC 0.572, train loss reduction 6%), so
> EXP-005 cannot fully separate capacity-bottleneck from
> backbone-bottleneck, but the observation that every predictor variant
> converges to the same DINOv2 cosine floor on this held-out family is
> the strongest evidence so far that the binding constraint is the
> frozen backbone, not the predictor class.

### 9.2 Numbers safe to quote

- MLP train AUROC: **0.572** [0.546, 0.595] (underfit training)
- MLP val AUROC: 0.293 [0.270, 0.315]
- MLP test AUROC: **0.270** [0.249, 0.293]
- DINOv2 ViT-B/14 baseline AUROC: 0.274 (MLP essentially matches this)
- Pixel L2 baseline AUROC: 0.580 (still strongest; unchanged from EXP-004)
- MLP fit runtime: 1.38 s on A10G (vs 22 s for numpy linear)
- Val / train loss ratio at end: **1.29** (vs EXP-004 linear's 2.07 —
  but only because train loss is also much higher)

### 9.3 Pedagogical beats

1. **"Copying the optimiser is not copying the training recipe."**
   The linear predictor's `weight - identity` regulariser and the
   MLP's PyTorch `weight_decay` (which pulls W toward zero) are not
   the same regularisation despite both being called "weight decay."
   Transferring hyperparameters across architectures deserves explicit
   justification.
2. **"Underfitting is a valid negative result."** EXP-005 looks like a
   failure but it is a clean demonstration that (a) you can't conclude
   "the MLP doesn't generalise" until you've first shown the MLP
   *fits*, and (b) a predictor that never leaves identity reveals how
   the frozen backbone alone scores the task.
3. **"Your JEPA predictor is only as good as where it starts."** The
   identity warm-start is a sensible choice when train-time targets
   are close to source, but it becomes an attractor that can't be
   escaped under certain optimiser configurations.
4. **"Write 'inconclusive' when inconclusive."** EXP-005 is not a
   confirmation that the backbone is the bottleneck. It is evidence
   *consistent with* that hypothesis, but the capacity-bottleneck
   alternative has not been ruled out. The report says so.

### 9.4 Updated cross-run table (the single most useful asset)

| Run | Predictor | Proxy | Train / val / test AUROC | Strongest baseline | Δ vs strongest |
|---|---|---|---:|---|---:|
| EXP-001 | linear | trivial | 0.999 / 1.000 / 1.000 | DINOv2-S = 1.000 | 0.000 |
| EXP-002 | linear | hardened, matched eval | 0.953 / 0.921 / 0.920 | DINOv2-S = 0.652 | **+0.269** |
| EXP-003 | linear | hardened, one-family held-out eval | 0.953 / 0.710 / 0.680 | SSIM = 0.961 | −0.281 |
| EXP-004 | linear | hardened, mixed train + third-family eval | 0.900 / 0.265 / 0.249 | Pixel L2 = 0.580 | −0.331 |
| **EXP-005** | **MLP** | **same as EXP-004** | **0.572 / 0.293 / 0.270** | Pixel L2 = 0.580 | **−0.310** |

### 9.5 Paper-section mapping

| Paper section | Drawn from EXP-005 |
|---|---|
| Predictor-class ablation | §2.1, §4.1, §4.2 |
| Training-configuration diagnostic | §4.3, §4.4 |
| Five-run joint reading | §5.2 |
| Thesis-level statement | §5.4 |
| Limitations | §6 |
| EXP-006 design | §7 |

### 9.6 Plot assets

- `artifacts/plots/baseline_score_histogram.png` — identical baselines
  to EXP-004 on `strong_held_out_2`.
- `artifacts/plots/jepa_score_histogram.png` — MLP test-split drift
  scores; compare to EXP-004's to see the near-identity pattern.
- `logs/progress.jsonl` — full per-stage timing trace.

### 9.7 Five-experiment arc update

The story now has a fifth act that is *more* honest because it
complicates the clean three-act EXP-001/002/003 narrative:

1. Build pipeline, proxy is trivial.
2. Harden proxy, JEPA wins +0.27.
3. Held-out family: win collapses.
4. Mixed-family training on third unseen family: predictor inverts
   below random.
5. MLP predictor on the same proxy: underfits training under the
   inherited optimiser; test AUROC matches frozen DINOv2 cosine. The
   next lever is to tune the MLP and swap the backbone — both scoped
   as EXP-006.

That progression is an accurate story about research: experiments
answer one sub-question each, some answers come with caveats, and the
caveats get addressed by the next experiment rather than hidden.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-24 | AbdelStark + Claude Code | Initial report; run completed on `47cbafe`. EXP-006 scope locked: Adam-tuned MLP + DINOv2→OpenCLIP backbone swap. |
