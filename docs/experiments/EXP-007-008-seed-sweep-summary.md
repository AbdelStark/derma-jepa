# EXP-007 + EXP-008 — seed-sweep summary

**Status:** Completed. Five seeds per configuration, ten runs total.
**Outcome:** Both headline numbers and the partition narrative survive seed variance. EXP-007 (DermLIP) test AUROC across 5 seeds is **0.9435 ± 0.0029**; EXP-008 (BiomedCLIP) test AUROC across 5 seeds is **0.3286 ± 0.0120**. The ratio of seed-to-seed std between the two configurations (0.012 / 0.003 ≈ 4×) is interpretable: the high-AUROC regime under DermLIP saturates predictor parameters more reproducibly across seeds than the partial-fit regime under BiomedCLIP. The three-way pretraining-data gradient (web 0.286 < general medical 0.329 < dermoscopy 0.944) is unchanged when the BiomedCLIP and DermLIP cells are populated with seed-mean point estimates.
**Date (UTC):** 2026-04-27
**Hardware:** Hugging Face Jobs, `a10g-large` × 5 seeds × 2 configs (parallel)
**Launcher:** `scripts/hf_jobs_seed_sweep.sh` (commit `c94bf4e`)
**Aggregator:** `scripts/aggregate_seed_sweep.py`

---

## 1. What this run did

EXP-007 reported test AUROC 0.9447 on `strong_held_out_2` for the DermLIP backbone; EXP-008 reported 0.3247 for BiomedCLIP. Both used a single seed (20260422, the project default since EXP-001). The seed determines lesion-ID split assignment, pair-generation pseudo-randomness, predictor weight initialisation, and the data-loader shuffle order. None of those choices is a priori expected to matter at this scale, but every prior run had used the same seed. Locking bootstrap-CI-tight ranges on the two existing headlines is cheap GPU time and worth doing before any larger follow-up.

Four additional seeds (1, 2, 3, 4) per configuration were launched in parallel via the new `hf_jobs_seed_sweep.sh` launcher. Each seed run uses an identical YAML config to its base (EXP-007 or EXP-008) with only `seed:` and `run_id:` substituted, then exec's the standard train bundle. Two of the eight initial launches failed at the HF "Volume mount failed" infra error (same as observed earlier on April 24); both were re-launched with no code change and completed cleanly. End-to-end wall time was ~85 min for the parallel cohort plus ~85 min for the two re-launches that hit the infra flake.

---

## 2. Per-seed AUROCs

### EXP-007 DermLIP (5 seeds)

| run_id | train | val | test |
|---|---:|---:|---:|
| `…dermlip-exp007-v1` (seed 20260422, original) | 0.9999 | 0.9435 | 0.9447 |
| `…dermlip-exp007-seed-1-v1` | 0.9999 | 0.9368 | 0.9470 |
| `…dermlip-exp007-seed-2-v1` | 0.9999 | 0.9424 | 0.9392 |
| `…dermlip-exp007-seed-3-v1` | 1.0000 | 0.9485 | 0.9444 |
| `…dermlip-exp007-seed-4-v1` | 0.9999 | 0.9508 | 0.9422 |

### EXP-008 BiomedCLIP (5 seeds)

| run_id | train | val | test |
|---|---:|---:|---:|
| `…biomedclip-exp008-v1` (seed 20260422, original) | 0.9226 | 0.3507 | 0.3247 |
| `…biomedclip-exp008-seed-1-v1` | 0.9077 | 0.3248 | 0.3436 |
| `…biomedclip-exp008-seed-2-v1` | 0.9133 | 0.3258 | 0.3358 |
| `…biomedclip-exp008-seed-3-v1` | 0.9043 | 0.3150 | 0.3119 |
| `…biomedclip-exp008-seed-4-v1` | 0.9143 | 0.3415 | 0.3269 |

---

## 3. Across-seed summary

| Configuration | Split | n | Mean | Std | Min | Max | 95% CI[mean] |
|---|---|---:|---:|---:|---:|---:|:---:|
| **EXP-007 DermLIP** | train | 5 | 0.9999 | 0.0000 | 0.9999 | 1.0000 | [0.9999, 1.0000] |
| EXP-007 DermLIP | val | 5 | 0.9444 | 0.0055 | 0.9368 | 0.9508 | [0.9396, 0.9492] |
| **EXP-007 DermLIP** | **test** | 5 | **0.9435** | **0.0029** | 0.9392 | 0.9470 | **[0.9409, 0.9461]** |
| EXP-008 BiomedCLIP | train | 5 | 0.9124 | 0.0070 | 0.9043 | 0.9226 | [0.9063, 0.9186] |
| EXP-008 BiomedCLIP | val | 5 | 0.3316 | 0.0143 | 0.3150 | 0.3507 | [0.3190, 0.3441] |
| **EXP-008 BiomedCLIP** | **test** | 5 | **0.3286** | **0.0120** | 0.3119 | 0.3436 | **[0.3181, 0.3391]** |

The 95 % CI[mean] on EXP-007's test AUROC is [0.9409, 0.9461] — narrower than the bootstrap CI of any single run and well above pixel L2's upper bound (0.606). The 95 % CI[mean] on EXP-008's test AUROC is [0.3181, 0.3391] — also narrow, below random, and below pixel L2.

---

## 4. Updated three-way pretraining-data table

Replacing the single-seed point estimates from EXP-006b / EXP-007 / EXP-008 with seed-mean point estimates where available:

| Run | Backbone | Pretraining | Test AUROC (seed mean) | Test AUROC std | n seeds | Δ vs strongest baseline |
|---|---|---|---:|---:|---:|---:|
| EXP-006b | OpenAI CLIP B/16 | LAION (web) | 0.286 | — | 1 | −0.294 |
| **EXP-008** | **BiomedCLIP B/16** | **PMC-15M (general medical)** | **0.329** | **0.012** | **5** | **−0.252** |
| **EXP-007** | **DermLIP B/16** | **Derm1M (dermatology)** | **0.944** | **0.003** | **5** | **+0.363** |

Web → general-medical step: **+0.04 AUROC** (single-seed CLIP vs 5-seed-mean BiomedCLIP).
General-medical → dermoscopy step: **+0.62 AUROC** (5-seed-mean BiomedCLIP vs 5-seed-mean DermLIP).
The non-uniformity in the gradient (~15× more lift in the dermoscopy step than in the medical-domain step), established by single-seed runs in EXP-008 §4.3, is preserved with seed-mean point estimates.

---

## 5. What this changes / does not change

**Confirms:** The single-seed headlines from EXP-007 (0.945) and EXP-008 (0.325) are not seed-sensitive. Seed-mean drift is small: −0.001 AUROC for EXP-007, +0.004 AUROC for EXP-008. Either headline could be quoted as "test AUROC ≈ 0.94 across 5 seeds, std 0.003" / "test AUROC ≈ 0.33 across 5 seeds, std 0.012" in any paper draft without further qualification.

**Confirms:** The seed-to-seed std on EXP-008 (0.012) is roughly 4× the std on EXP-007 (0.003). Plausible read: the high-AUROC regime under DermLIP saturates the linear predictor's ranking near the ceiling of what the embedding space encodes, so seed-driven variations in the predictor's weight initialisation and split assignment have small effect on the resulting test ranking. Under BiomedCLIP, the predictor sits in the partial-fit regime where small ranking shuffles on test pairs translate more directly to AUROC variance. This is consistent with what one would expect from a Brier-score-like sensitivity argument and does not surface anything anomalous.

**Does not change:** EXP-009 priority. The EXP-007 win still cannot be partitioned between (a) dermoscopy-domain transfer in Derm1M's non-HAM10000 portion and (b) HAM10000 image-level overlap. Seed sweeps cannot address that question; only a non-HAM10000 dermoscopy pretrain can. EXP-009's design from EXP-008 §7 stands.

**Does not change:** Limitations from EXP-007 §6 and EXP-008 §6. The contamination caveat, the single-architecture restriction, the synthetic-nuisance design, and the cross-sectional-HAM10000 caveat all apply unchanged.

---

## 6. Headline numbers safe to quote in paper draft

- DermLIP-linear test AUROC: **0.944 ± 0.003** (n = 5 seeds, range 0.939–0.947, 95 % CI[mean] [0.941, 0.946]).
- BiomedCLIP-linear test AUROC: **0.329 ± 0.012** (n = 5 seeds, range 0.312–0.344, 95 % CI[mean] [0.318, 0.339]).
- OpenAI CLIP-linear test AUROC: 0.286 (n = 1 seed, single run; further seed sweeps not run because the central paper claim does not depend on web-CLIP variance).
- Pixel L2 baseline AUROC: 0.580 (deterministic, no seed dependence).
- Web → general-medical step: **+0.04 AUROC** (lift small).
- General-medical → dermoscopy step: **+0.62 AUROC** (lift dominant).

---

## 7. Reproducibility

### 7.1 Launch (per seed)

```bash
BASE_CONFIG=configs/data/ham10000_hf_mounted_exp007.yaml \
SEED=$SEED \
SWEEP_TAG=dermlip-exp007 \
HF_JOBS_DETACH=1 \
  ./scripts/hf_jobs_seed_sweep.sh
```

(Replace `BASE_CONFIG` and `SWEEP_TAG` for EXP-008.)

### 7.2 Aggregate

```bash
uv run python scripts/aggregate_seed_sweep.py \
  --label "EXP-007 DermLIP" \
  --run-id ham10000-hf-dermlip-exp007-v1 \
  --run-id ham10000-hf-dermlip-exp007-seed-1-v1 \
  --run-id ham10000-hf-dermlip-exp007-seed-2-v1 \
  --run-id ham10000-hf-dermlip-exp007-seed-3-v1 \
  --run-id ham10000-hf-dermlip-exp007-seed-4-v1
```

Local mirrors of `metrics.json`, `baseline_metrics.json`, and `config.yaml` for each of the 10 seed-sweep runs are pulled directly from the HF dataset by the script when not present locally.

---

## 8. What's next

EXP-009 is now unblocked. Per EXP-008 §7, the highest-information next experiment is **self-pretraining DINOv2 ViT-B/14 on a non-HAM10000 dermoscopy corpus** (ISIC archives minus HAM10000 split, DermNet, BCN20000 non-HAM10000 components, …) for a short JEPA-style or MIM objective and re-running the EXP-004 recipe on top. That is the only experiment that partitions dermoscopy-domain transfer from HAM10000 contamination, which is currently the central caveat on the EXP-007 paper claim.

A seed sweep on EXP-009 itself (matching this protocol — 5 seeds, parallel) should be planned into its launch, since the partition question is paper-headline-relevant and a single-seed result there would not be sufficient.

---

## 9. Changelog

| Date (UTC) | Author | Change |
|---|---|---|
| 2026-04-27 | AbdelStark + Claude Code | Initial summary. 5 seeds per configuration, 10 runs total. Both headlines and the partition gradient survive seed variance with std 0.003 (EXP-007) / 0.012 (EXP-008). EXP-009 (non-HAM10000 dermoscopy SSL pretrain) unblocked. |
