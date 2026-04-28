# EXP-001 — HAM10000 primary-tier JEPA run (`ham10000-hf-dinov2-primary-v1`)

**Status:** Completed, uploaded, analyzed.
**Outcome:** Ceiling result + proxy-task signal (see §5 Analysis).
**Date (UTC):** 2026-04-22
**Hardware:** Hugging Face Jobs, `a10g-large` (1× NVIDIA A10G 24 GB)
**Run location:** `hf://datasets/abdelstark/derma-jepa-runs/ham10000-hf-dinov2-primary-v1`
**Local mirror:** `outputs/hf-runs/ham10000-hf-dinov2-primary-v1/`
**Launcher commit:** `e92dbc5` (pre-observability; next run will be on `d93d72e`)

---

## 1. Summary

The first primary-tier HAM10000 JEPA-style training run finished cleanly on
Hugging Face Jobs. The JEPA-style latent predictor reached **0.9998 AUROC**
on the test split with a tight 95% bootstrap CI of [0.9996, 1.0000].

The strongest cheap baseline is **frozen DINOv2 ViT-S/14 cosine distance**
at **0.99995 AUROC** [0.9999, 1.0000]. The JEPA predictor therefore landed
**−0.0001** versus the strongest baseline — statistically indistinguishable,
both at ceiling.

The important finding is not the JEPA delta. It is that **every dense
baseline is also at ceiling on this proxy task**: even pixel L2 scores
0.997 AUROC. That reveals a property of the proxy-task construction, not a
property of the JEPA objective. The honest next step is to harden the
proxy before interpreting any model comparison.

Collapse checks pass (prediction norm = 1.0, min dimension variance ≠ 0).
This is a legitimate result, not a training bug.

---

## 2. Experimental setup

### 2.1 Thesis under test

From `docs/spec/MVP-SPEC.md`:

> DermaJEPA evaluates whether a JEPA-style latent trajectory model can
> separate nuisance-induced visual drift from meaningful lesion-change
> proxies better than simple pixel/SSIM baselines and generic frozen vision
> embeddings on public dermatology data.

### 2.2 Dataset

**Source:** HAM10000 (Tschandl et al., *Scientific Data*, 2018) — 10,015
dermoscopic images across 7 diagnoses. Uploaded as a private Hugging Face
dataset at `hf://datasets/abdelstark/ham10000` for the mounted-volume path;
repo never vendored.

Diagnosis distribution in the normalized metadata:

| Dx | Count |
|---|---:|
| nv (melanocytic nevus) | 6,705 |
| mel (melanoma) | 1,113 |
| bkl (benign keratosis-like) | 1,099 |
| bcc (basal cell carcinoma) | 514 |
| akiec (actinic keratosis / IEC) | 327 |
| vasc (vascular) | 142 |
| df (dermatofibroma) | 115 |

Anatomical sites span abdomen, back, chest, ear, face, foot, genital, hand,
lower extremity, neck, scalp, trunk, and upper extremity.

**Known data limitation:** HAM10000 provides no patient identifier
(`patient_id_present = 0`). The audit therefore falls back to `lesion_id`
as the leakage boundary for splits.

Audit artifact: `data_audit.json` in the run directory.

### 2.3 Proxy-task construction

Two pair classes per split:

- **Stable pair** — the source image, paired with a deterministic mild
  nuisance variant generated after split assignment. Variants are mild
  brightness / saturation / rotation perturbations written under
  `public/stable_variants/<split>/<image_id>_stable_<pair_index>.png`.
- **Changing pair** — two different lesion images chosen by falling back
  through same-patient → same-diagnosis-and-site → same-diagnosis →
  same-site → any match. Positive class = `changing`.

1,000 stable + 1,000 changing pairs per split (train/val/test) → **6,000
pairs total**.

### 2.4 Splits and leakage

Splits are deterministic, seeded at 20260422, grouped by lesion ID because
patient ID is not present in HAM10000:

| Split | Groups (lesions) | Pairs | Stable | Changing |
|---|---:|---:|---:|---:|
| train | 5,229 | 2,000 | 1,000 | 1,000 |
| val | 1,120 | 2,000 | 1,000 | 1,000 |
| test | 1,121 | 2,000 | 1,000 | 1,000 |

Leakage probes report **zero** lesion-ID overlap and **zero** source-dataset
overlap across splits. Two duplicate image-checksum probes fired on the
metadata (a known HAM10000 artifact), with no duplicate image IDs.

### 2.5 Models and objective

**Frozen embedding models** (DINOv2 backbones from Hugging Face):

- `dinov2_vits14` — `facebook/dinov2-small`, CLS token, batch 32
- `dinov2_vitb14` — `facebook/dinov2-base`, CLS token, batch 16

**JEPA-style predictor:** a linear map from DINOv2 ViT-B/14 context
embeddings to target embeddings, initialized at identity + small noise.
Optimizes MSE between `W x_context + b` and `x_target` with L2 weight
decay toward identity. Trained on **stable pairs only** (N = 1,000) for
200 epochs, batch 128, LR 0.03, weight decay 0.001.

**Score at test time:** cosine distance between predicted and target
latent. Positive class is `changing`, so higher score → more likely
changing.

### 2.6 Baselines

| Baseline | Description |
|---|---|
| pixel_l2 | L2 distance over 224-pixel preprocessed RGB images |
| ssim_distance | `1 − SSIM` over greyscale 224-pixel images |
| embedding_cosine_dinov2_vits14 | `1 − cos(x_context, x_target)` over DINOv2 ViT-S/14 |
| embedding_cosine_dinov2_vitb14 | Same with DINOv2 ViT-B/14 |

### 2.7 Metrics

Primary: **AUROC** over the test split with 1,000-sample bootstrap 95% CI.
Supporting: AUPRC, equal-error-rate threshold, FPR at fixed TPR=0.80.
Config: `bootstrap_samples=1000`, `ci_level=0.95`, `fixed_tpr=0.80`.

### 2.8 Infrastructure snapshot

- Image: `ghcr.io/astral-sh/uv:python3.12-...` (HF Jobs `uv run` runtime)
- Python: 3.12.12
- Platform: Linux-6.12.79-101.147.amzn2023.x86_64 (glibc 2.36)
- torch 2.11.0, transformers 5.5.4, timm 1.0.26, numpy 2.4.4 (all pinned
  via `scripts/hf_jobs_constraints.txt`)
- CUDA: cu13 stack (cuBLAS, cuDNN 9.19.0, NCCL 2.28.9, Triton 3.6.0)
- Dataset mounted read-only at `/data` via `hf://datasets/abdelstark/ham10000:/data:ro`

Environment snapshot: `environment.txt` in the run directory.

---

## 3. Operational timeline

Wall-clock phases from the Job log and `logs/*.log` timestamps:

| Phase | Start (UTC) | Duration | Notes |
|---|---|---|---|
| Scheduling | 13:39:37 | ≤ 2 min | Waited once on A100, fell back to A10G |
| Dependency install | ≈ 13:40 | ~5 min | ~3.5 GB of wheels; pinned numpy/scipy/torch/transformers |
| Manifest build | | ≤ 1 min | 10,015 HAM10000 rows validated through FUSE mount (first-touch reads) |
| Manifest write | 14:49:29 | | `built public proxy manifest with 6000 pairs` |
| Embedding export | ≈ 14:49 | ~32 min | DINOv2 ViT-S/14 + ViT-B/14, 10,015 images × 2 passes over Hub-backed FUSE |
| Embedding end | 15:21:35 | | `exported 2 embedding model(s) for 8015 images` |
| Baselines | 15:21–15:26 | ~5 min | pixel L2, SSIM, embedding cosines |
| Baseline end | 15:26:41 | | `evaluated public baselines on test split` |
| JEPA fit | 15:26:41–15:27:20 | **28.6 s** | 200 epochs × 1,000 pairs, linear predictor |
| Upload | ≈ 15:27–15:30 | ~3 min | 3,014 files, ~295 MB to `abdelstark/derma-jepa-runs` |
| **Total wall time** | | **≈ 1h 48m** | A10G sat idle during install/upload |

Note that `embed.log` reports `8015 images` because
`_unique_images(manifest_rows)` counts distinct image IDs in the 6,000-pair
manifest, not the 10,015 raw metadata rows. The embedding NPZ files are
the authoritative source.

### 3.1 Observability gap (fixed for next run)

This run predates `d93d72e` so the only progress signals during the 32-min
embedding export were two lines from `transformers`:

```
Loading weights: 100%|██████████| 223/223 [00:00<00:00, 3485.72it/s]
Loading weights: 100%|██████████| 223/223 [00:00<00:00, 3386.30it/s]
```

That created a ~30-minute silent stretch where it was impossible to tell
a healthy Job from a stuck one. The follow-up commit added
`src/derma_jepa/observability.py`, which emits structured stage start/end
events and progress ticks to stdout and to `<run_dir>/logs/progress.jsonl`.
EXP-002 onwards will include that trace directly in the run directory.

### 3.2 Cost envelope

`a10g-large` for ~1h 48m; dominated by embedding export over the FUSE-mounted
dataset, not by GPU compute. A warm-cached rerun of the same config could
plausibly be 15–25 minutes because DINOv2 weights are cached and the
compute-bound stages are ~1 minute combined.

---

## 4. Results

### 4.1 Headline numbers (test split, N = 2,000 pairs)

| Model / baseline | AUROC | 95% CI | AUPRC | EER | Δ vs strongest |
|---|---:|:---:|---:|---:|---:|
| **JEPA predictor (v1)** | **0.9998** | [0.9996, 1.0000] | 0.9998 | 0.0055 | −0.0001 |
| DINOv2 ViT-S/14 cosine | 0.99995 | [0.9999, 1.0000] | 0.99995 | 0.0050 | — |
| DINOv2 ViT-B/14 cosine | 0.9997 | [0.9994, 0.9999] | 0.9997 | 0.0065 | −0.0002 |
| Pixel L2 | 0.9968 | [0.9949, 0.9984] | 0.9963 | 0.0230 | −0.0032 |
| SSIM distance | 0.8955 | [0.8815, 0.9088] | 0.8690 | 0.1860 | −0.1044 |

At fixed TPR = 0.80, all dense baselines and the JEPA predictor achieve
FPR = 0.000 — the operating point is trivially in the linearly-separable
region.

### 4.2 JEPA across splits

| Split | AUROC | 95% CI | EER |
|---|---:|:---:|---:|
| train | 0.99985 | [0.99965, 0.99998] | 0.0040 |
| val | 0.99996 | [0.99988, 1.00000] | 0.0015 |
| test | 0.99981 | [0.99959, 0.99997] | 0.0055 |

Near-identical across splits; the gap between train and test is a rounding
error. No overfitting signal.

### 4.3 Training dynamics

Losses barely move — identity initialization over L2-normalized DINOv2
vectors is already an excellent predictor of stable-pair targets.

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 0.000092 | 0.000084 |
| 20 | 0.000083 | 0.000077 |
| 100 | 0.000083 | 0.000076 |
| 200 | 0.000082 | 0.000075 |

Δ(val_loss, 1→200) ≈ 9 × 10⁻⁶. Linear predictor fit took 28.6 seconds.

### 4.4 Representation health

- `prediction_norm_mean` = 1.0 (exact, L2 normalized predictor output)
- `prediction_norm_min` = 0.9999998
- `dimension_variance_mean` = 4.9 × 10⁻⁴
- `dimension_variance_min` = 1.7 × 10⁻⁵
- `collapsed` = **False**

Predictions preserve the DINOv2 unit-sphere structure and retain
per-dimension variance across the test set.

### 4.5 Pair-score distributions (test split)

| Score | Stable mean | Stable p95 | Stable max | Changing mean | Changing p5 | Changing min |
|---|---:|---:|---:|---:|---:|---:|
| pixel_l2 | 0.044 | 0.077 | 0.182 | 0.173 | 0.100 | 0.070 |
| ssim_distance | 0.226 | 0.465 | 0.851 | 0.449 | 0.272 | 0.210 |
| DINOv2-S cos | 0.026 | 0.054 | 0.110 | 0.303 | 0.142 | 0.070 |
| DINOv2-B cos | 0.029 | 0.061 | 0.162 | 0.324 | 0.141 | 0.059 |
| JEPA (test) | 0.029 | 0.060 | 0.161 | 0.329 | 0.145 | 0.068 |

For every dense score, the distance between stable-class mean and
changing-class mean is an order of magnitude, with only a tiny overlap in
the tails.

### 4.6 Plots in the run directory

- `artifacts/plots/baseline_score_histogram.png`
- `artifacts/plots/jepa_score_histogram.png`

---

## 5. Analysis

### 5.1 Every dense score is at ceiling

Pixel L2 reaches **0.997 AUROC** on this proxy. That is decisive.

Pixel L2 has no notion of lesion biology, patient identity, or clinical
context. For pixel L2 to score this well, the "stable" and "changing"
pair classes must already be trivially separable in raw pixel space.

Looking at the construction:

- Stable pairs are `(source_image, source_image + mild nuisance
  variant)` — i.e. the *same* underlying photograph with small color /
  geometry perturbation.
- Changing pairs are two *different* photographs of different lesions.

Under that construction, any reasonable distance between two pixel
matrices will be small for stable pairs (same image ± noise) and large for
changing pairs (different images). Pixel L2 happily exploits this because
the stable-vs-changing decision reduces to "is the second image the same
image as the first".

SSIM is the only score that *isn't* near ceiling (0.896), because SSIM
normalizes for mean/contrast and captures structural similarity more
tolerantly than L2 — so nuisance perturbations on the same image shift
SSIM more than they shift L2.

### 5.2 What this means for the JEPA delta

Because the proxy is trivially separable, the JEPA predictor has almost no
room to *differentiate* itself from frozen DINOv2. The measured delta
(−0.0001) does **not** mean "JEPA does not help for lesion monitoring." It
means "on this proxy, any competent encoder ties with any other competent
encoder, and the linear predictor can neither lose nor win meaningfully."

This is explicitly anticipated in `docs/spec/MVP-SPEC.md` §21 as a research
risk:

> **Risk:** proxy task is too synthetic.
> **Check:** compare performance on mild nuisance pairs, visually similar
> changing negatives, and PAD-UFES-20 stress cases separately.

EXP-001 is the first hard confirmation of that risk.

### 5.3 Where the small amount of difficulty sits

The failure-case report (`artifacts/reports/jepa_failure_cases.json`)
surfaces exactly the two regimes where the task stops being trivial:

**Hardest stables** (stable pairs with high predicted drift):

| Score | Diagnosis | Site | Construction reason |
|---:|---|---|---|
| 0.161 | mel | upper extremity | same_image_post_split_mild_nuisance |
| 0.122 | nv | lower extremity | same_image_post_split_mild_nuisance |
| 0.120 | bcc | chest | same_image_post_split_mild_nuisance |

These are cases where the deterministic nuisance variant happens to move
the DINOv2 embedding unusually far from its source — a sensitivity story,
not a true lesion-change story.

**Hardest changings** (changing pairs with low predicted drift):

| Score | Diagnosis | Site | Construction reason |
|---:|---|---|---|
| 0.068 | bkl | abdomen | different_lesion_same_diagnosis_and_site |
| 0.069 | mel | neck | different_lesion_same_diagnosis_and_site |
| 0.074 | nv | trunk | different_lesion_same_diagnosis_and_site |

These are the *right* kind of hard case: two genuinely different lesions
that share diagnosis and anatomical site, so DINOv2 embeds them close.
Future proxy designs should over-weight this regime — it is where any
claim of "lesion-aware representation" actually has to live.

### 5.4 What the ceiling does not invalidate

Even though the headline AUROCs are not interpretable as a JEPA win,
several properties remain defensible and carry into the paper / writeups:

- **Pipeline correctness.** Manifest build, leakage probes, embedding
  export, baselines, JEPA fit, representation-health checks, artifact
  upload, and `hf-run summary` all worked end-to-end on 10k real images.
- **No representation collapse.** `dimension_variance_min = 1.7e−5`,
  norms preserved at 1.0. The JEPA objective is not training a degenerate
  encoder.
- **Consistent cross-split behaviour.** No train/val/test anomaly; lesion-
  ID-level splitting is producing similar-difficulty held-out data.
- **Honest negative-result reporting is working.** `metrics.json` and the
  model card already emit the spec-compliant phrasing
  ("JEPA-style predictor did not beat the strongest baseline on this
  split; report as a legitimate negative or inconclusive result").

---

## 6. Limitations and threats to validity

1. **Proxy is too easy.** See §5.1. This is the dominant limitation.
2. **No patient IDs in HAM10000.** Splits are lesion-ID-level, not
   patient-level. If lesion IDs are reused across photographs of the same
   patient, there is residual patient-level leakage risk the audit cannot
   detect.
3. **Stable nuisance variants are synthetic.** Real lesion re-photography
   introduces illumination, angle, skin-texture, hair, camera-quality, and
   framing variation that the mild color/geometry perturbations used here
   do not approximate.
4. **Changing pairs are not genuinely longitudinal.** They are two
   different lesions drawn from overlapping clinical categories. This is
   explicitly the "longitudinal-proxy" compromise HAM10000 forces —
   HAM10000 is a cross-sectional archive with no repeated-subject
   structure.
5. **JEPA predictor is a single linear map from identity.** The MVP spec
   allows this for the scaffolded run; any claim about the JEPA *objective*
   from this run alone would be premature.
6. **Single run, single seed.** No multi-seed aggregation; the tight
   bootstrap CIs here are over pair resampling, not over training seeds.
7. **Observability gap during embedding export.** EXP-001 had no per-batch
   progress stream. Fixed on `main` at `d93d72e` for EXP-002.

---

## 7. What changes for the next run

Concrete, bounded follow-ups that preserve the spec-locked thesis:

1. **Strengthen the proxy task.** Priority #1. Two options:
   - **Harder stables:** swap the post-split nuisance variants for a
     stronger nuisance family (lighting + angle + partial crop + moderate
     compression) so pixel L2 no longer separates classes.
   - **Visually-constrained changings:** restrict changing pairs to the
     same-diagnosis-and-site bucket only (the current fallback) and drop
     the looser fallbacks. Failure-case data already shows this is the
     regime where any real "lesion-aware" signal must live.
2. **Add PAD-UFES-20 as an out-of-distribution stress test.** Same
   encoder, same predictor, smartphone / clinical capture. If JEPA still
   ties pixel L2 there, the thesis is genuinely in trouble on
   cross-sectional data.
3. **Seed sweep.** 3 seeds on the primary config; report mean ± std of the
   delta, not a single point estimate.
4. **Report at fixed-FPR operating point.** Add FPR = 0.01 and FPR = 0.05
   alongside the current fixed-TPR row for a more demanding threshold
   readout.
5. **Use the new observability stream as a diagnostic.** Pull the
   `logs/progress.jsonl` from EXP-002 to build an apples-to-apples timing
   table instead of inferring it from log timestamps.

Explicitly **not** doing:

- Tuning the JEPA predictor hyperparameters to try to scrape a positive
  delta. The MVP spec §20 forbids threshold tuning or baseline weakening
  to rescue a negative result.
- Branching out into segmentation / SAM / diagnosis classifiers. See the
  ml-intern research spike.
- Scaling the predictor to a deeper MLP or transformer yet. Proxy-task
  design dominates; bigger models on a trivial task teach us nothing new.

---

## 8. Reproducibility

### 8.1 Launch command

```bash
unset HF_JOBS_DRY_RUN HF_JOBS_FLAVOR HF_JOBS_TIMEOUT HF_JOBS_DETACH
export DERMA_JEPA_CONFIG_PATH=configs/data/ham10000_hf_mounted.yaml
export DERMA_JEPA_INSTALL_EXTRAS=model
export HF_JOBS_VOLUME="hf://datasets/abdelstark/ham10000:/data:ro"
export HF_JOBS_FLAVOR=a10g-large
export HF_JOBS_TIMEOUT=12h
export HF_JOBS_DETACH=1
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-primary-v1 \
  ./scripts/hf_jobs_ham10000_primary.sh
```

### 8.2 Pull and verify

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id abdelstark/derma-jepa-runs \
  --run-id ham10000-hf-dinov2-primary-v1
```

Expected top line: `auroc: 0.9998`, `strongest_baseline: embedding_cosine_dinov2_vits14 = 1.0000`, `delta_vs_baseline: -0.0001`, `collapsed: False`.

### 8.3 Artifact contract (uploaded)

- `config.yaml` — resolved primary-tier HAM10000 config
- `environment.txt` — Python, platform, timestamp
- `data_audit.json` — metadata coverage, leakage probes, duplicate checks
- `metadata_normalized.parquet` — 10,015 normalized records
- `manifest_{all,train,val,test}.parquet` — 6,000-pair proxy manifest
- `baseline_metrics.json` — all four baselines with CIs and full pair scores
- `metrics.json` — primary, per-split, and representation-health numbers
- `model_card.md` — spec-compliant card with the negative-result wording
- `artifacts/embeddings/` — DINOv2 ViT-S/14 and ViT-B/14 NPZ + parquet,
  plus JEPA predicted-latent NPZ / parquet
- `artifacts/models/jepa_predictor.npz` — linear predictor weights + bias
- `artifacts/plots/{baseline,jepa}_score_histogram.png`
- `artifacts/reports/{baseline,jepa}_failure_cases.json`
- `artifacts/reports/gold_audit_subset.csv`
- `public/stable_variants/{train,val,test}/*.png` — 3,000 deterministic
  nuisance-variant images used as the stable-pair target
- `logs/{manifest,embed,eval,train}.log`

---

## 9. Assets for future writeups

Pre-extracted, citation-ready material for articles, blog posts, videos,
conference talks, and the eventual paper. Everything here is grounded in
the artifacts above — do not cite numbers that are not in this file or in
the run directory.

### 9.1 Quotable headline

> On a leakage-controlled HAM10000 longitudinal-proxy task with 6,000
> stable / changing image pairs, a JEPA-style linear latent predictor over
> frozen DINOv2 ViT-B/14 reached 0.9998 AUROC on the held-out test split
> (95% bootstrap CI [0.9996, 1.0000]). The strongest cheap baseline —
> frozen DINOv2 ViT-S/14 cosine distance — reached 0.99995 AUROC. Under
> our construction, pixel L2 also cleared 0.997 AUROC, revealing that the
> proxy task as currently specified is trivially separable. The measured
> ceiling is a property of the task, not of the JEPA objective.

### 9.2 Headline numbers (safe to quote)

- 10,015 HAM10000 images audited end to end
- 6,000 stable / changing pairs with zero lesion-ID overlap between splits
- JEPA AUROC = 0.9998 [0.9996, 1.0000]
- Strongest baseline AUROC = 0.99995 [0.9999, 1.0000]
- Pixel L2 AUROC = 0.997; SSIM AUROC = 0.896
- JEPA Δ vs strongest baseline = −0.0001 (not significant)
- Collapse checks passed; dimension variance min = 1.7 × 10⁻⁵
- Linear-predictor fit time = 28.6 s on 1× A10G
- End-to-end Job wall time ≈ 1h 48m

### 9.3 Pedagogical beats

Points this run illustrates that transfer well to talks and educational
content:

1. **"Your strongest baseline is probably your dataset."** The most
   common source of a fake positive result isn't a model trick; it's a
   proxy task where pixel L2 quietly solves the decision.
2. **"Report the ceiling, don't paper over it."** The MVP spec's
   negative-result wording protects against spin. The fact that the
   written model card, `metrics.json`, and `hf-run summary` all say the
   same honest sentence is a design win, not a bug.
3. **"Observability is a research-quality issue."** A 30-minute silent
   embedding-export window cost real debugging time. Progress events
   aren't DevOps polish; they are how you distinguish "stuck" from
   "slow-but-healthy" in a hosted job.
4. **"Leakage audits are load-bearing."** Without the patient/lesion/
   source probes in `data_audit.json`, a near-ceiling result on HAM10000
   would have been hard to defend as even plausibly leakage-free.

### 9.4 Plot assets (embeddable)

- `artifacts/plots/baseline_score_histogram.png` — stable-vs-changing
  score distributions for the four cheap baselines.
- `artifacts/plots/jepa_score_histogram.png` — same view for the JEPA
  predictor.

### 9.5 Paper-section mapping

If this run feeds into a methodology paper, the following mapping is
faithful to `docs/spec/MVP-SPEC.md`:

| Paper section | Drawn from |
|---|---|
| Dataset / audit | `data_audit.json`, §2.2, §2.4 |
| Proxy-task construction | §2.3 and `public/stable_variants/` |
| Model and predictor | §2.5 |
| Baselines | §2.6, §4.1 |
| Metric protocol | §2.7 (bootstrap CI, fixed TPR) |
| Primary result | §4.1, §4.2 |
| Representation health | §4.4 |
| Limitations | §6 |
| Failure analysis | §5.3 |
| Ops / reproducibility | §3, §8 |

### 9.6 Conference / talk hooks

- Demo: run `./scripts/hf_jobs_ham10000_primary.sh` live on stage;
  surface the `hf-run summary` terminal output when the Job lands.
- Narrative arc: "we built the whole MVP pipeline, ran it on real
  HAM10000, and the first thing it told us was to change the question" —
  the kind of story ML talks often *don't* get to tell in public.
- Slide-ready table: §4.1 fits verbatim into a single slide.

---

## 10. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-22 | Abdelhamid Bakhta | Initial report; run completed and analyzed. |
