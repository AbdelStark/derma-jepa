# EXP-009 plan - non-HAM10000 dermoscopy pretraining partition

**Status:** Planned, not yet launched.
**Purpose:** Partition the EXP-007 DermLIP positive into dermoscopy-domain transfer versus HAM10000 image-level overlap.
**Primary decision:** Does a vision encoder self-pretrained on dermoscopy or skin imagery that explicitly excludes HAM10000 recover the EXP-007 high-AUROC regime when plugged into the unchanged EXP-004 proxy?
**Current blocker:** The repository has hosted training and HAM10000 probe infrastructure, but it does not yet have corpus-audit tooling or a self-supervised pretraining command for EXP-009.
**Pricing source:** Hugging Face Jobs pricing, checked 2026-04-29: <https://huggingface.co/docs/hub/jobs-pricing>.

---

## 1. Scientific question

EXP-007 reaches test AUROC about 0.944 across five seeds with a frozen DermLIP backbone. DermLIP was trained on Derm1M, whose public descriptions make HAM10000 inclusion likely. EXP-008 replaces DermLIP with BiomedCLIP, a general-biomedical backbone with no documented raw HAM10000 / ISIC archive ingestion, and lands near 0.329 across five seeds.

The remaining ambiguity is narrow but load-bearing:

- **Hypothesis A, dermoscopy transfer:** DermLIP succeeds because dermoscopy-rich pretraining aligns nuisance directions across the three proxy nuisance families.
- **Hypothesis B, HAM10000 overlap:** DermLIP succeeds because its pretraining corpus included the same HAM10000 lesion images that the proxy evaluates, even though it did not see the synthetic nuisance augmentations.

EXP-009 tests Hypothesis A directly. It self-pretrains a DINOv2-style ViT-B/14 encoder on a clean non-HAM10000 dermoscopy/skin corpus, freezes that encoder, exports HAM10000 embeddings, and runs the unchanged EXP-004 linear predictor protocol on top.

The experiment must not use EXP-004 AUROC for checkpoint selection. Checkpoint selection happens only on non-HAM10000 pretraining validation metrics and collapse checks. The HAM10000 proxy is touched once, after the encoder is frozen.

---

## 2. Precommitted decision table

Use the EXP-004 `strong_held_out_2` test split and the same AUROC convention as prior reports.

| EXP-009 test AUROC | Interpretation | Paper-claim effect |
|---:|---|---|
| `>= 0.85` | Clean dermoscopy-domain pretraining is sufficient under this scaffold. | EXP-007 caveat downgrades from "unpartitioned" to "tested against clean dermoscopy pretraining." |
| `0.50-0.80` | Partial transfer. Clean dermoscopy helps, but does not reproduce DermLIP's ceiling. | Claim becomes "dermoscopy-domain pretraining helps; HAM10000 overlap or DermLIP-specific ingredients likely explain the remaining gap." |
| `0.30-0.50` | Clean dermoscopy pretraining behaves like the non-DermLIP backbones. | EXP-007 should be interpreted primarily as dataset-overlap or DermLIP-specific memorization/fit, not as demonstrated out-of-distribution dermoscopy transfer. |
| `< 0.30` | The clean pretrain is worse than BiomedCLIP/OpenAI-CLIP under this proxy, or the pretraining run failed. | Do not over-interpret until pretrain-health gates are rechecked. A low score is evidence against domain transfer only if pretrain quality passed. |

Report both the single primary seed and a five-seed probe sweep. A single-seed EXP-009 result is not strong enough for the paper headline.

---

## 3. Required implementation before launch

The existing HF Jobs path can train the current JEPA predictor from a bundled wheel, but EXP-009 needs two new surfaces before GPU spend:

| Needed surface | Proposed path | Purpose |
|---|---|---|
| Corpus audit command | `derma-jepa corpus audit --config configs/pretrain/exp009_corpus.yaml` | Build a clean manifest, compute HAM10000 exclusions, write audit artifacts. |
| SSL pretraining command | `derma-jepa pretrain --config configs/pretrain/exp009_dinov2_vitb14_non_ham10000.yaml` | Train the clean encoder and export checkpoints/metrics. |
| Encoder export integration | `embeddings.kind: local_dinov2_checkpoint` | Use the EXP-009 checkpoint in the existing HAM10000 embedding/probe pipeline. |
| HF audit launcher | `scripts/hf_jobs_exp009_corpus_audit.sh` | Run the corpus audit on hosted CPU or GPU with one mounted input dataset. |
| HF pretrain launcher | `scripts/hf_jobs_exp009_pretrain.sh` | Launch smoke, pilot, and full SSL pretraining jobs through the private-repo-safe bundle pattern. |
| Probe config | `configs/data/ham10000_hf_mounted_exp009.yaml` | Re-run EXP-004 with only `embedding_model_id` changed to the EXP-009 encoder. |
| Probe launcher | `scripts/hf_jobs_ham10000_exp009.sh` | Launch the primary EXP-009 probe. |
| Seed sweep | reuse `scripts/hf_jobs_seed_sweep.sh` | Run seeds 1-4 after the primary probe completes. |

All launchers should follow the existing private-repo-safe pattern: build a local wheel, embed the wheel and config into the Job script, and avoid cloning GitHub inside the Hugging Face Job.

The existing `scripts/hf_jobs_train_bundle.sh` accepts one `HF_JOBS_VOLUME`. Keep EXP-009 inputs in one mounted dataset repo to avoid first changing the launcher:

```text
hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro
```

If licensing prevents raw images from being stored in that dataset, the input repo should hold URL manifests plus checksums, and the audit job should download into ephemeral job storage. Do not commit or publish raw image files from the experiment unless the source license explicitly permits redistribution.

---

## 4. Data plan

### 4.1 Input dataset layout

Use a private or gated Hugging Face dataset for inputs:

```text
derma-jepa-exp009-inputs/
  README.md
  source_manifests/
    isic_components.csv
    dermnet.csv
    dermquest.csv
    pad_ufes_20.csv
    seven_point.csv
    ph2.csv
  raw_or_cache/
    <source_name>/...
  ham10000_reference/
    ham10000_hashes.parquet
    ham10000_ids.csv
  manifests/
    exp009_candidates.parquet
```

The committed repository should only contain configs, scripts, schemas, and reports. It should not contain raw medical images.

### 4.2 Candidate sources

Treat all sources as candidates until the audit passes. Source inclusion is by per-image provenance, not by dataset name alone.

| Tier | Source class | Examples | Inclusion rule |
|---|---|---|---|
| A | Dermoscopy-first public sources | ISIC components not identified as HAM10000, BCN20000 components, MSK-1..5, UDA, PH2, 7-point checklist | Include only images with source provenance, license status, SHA-256, and no HAM10000 exact or near-duplicate match. |
| B | Clinical skin image sources | DermNet NZ, DermQuest-derived public images, PAD-UFES-20 | Include as domain-broadening data. Keep source tags so the final report can separate dermoscopy-heavy and mixed-skin variants. |
| C | Ambiguous or weak-provenance sources | Any scrape or archive with missing IDs, missing license, or unclear collection origin | Exclude by default. Move to manual review only if corpus size is otherwise insufficient. |

Do not assume that "ISIC but not named HAM10000" is clean. Some ISIC tasks and aggregate archives have duplicate images, rehosted images, or derived metadata. The audit decides.

### 4.3 Manifest schema

The clean manifest should be a Parquet table with at least:

| Column | Meaning |
|---|---|
| `image_id` | Stable experiment-local image identifier. |
| `source_dataset` | Source collection name. |
| `source_record_id` | Original dataset or archive ID, if available. |
| `source_uri` | Original URL or dataset path. |
| `license` | SPDX-like license string or explicit source license note. |
| `local_path` | Path inside the mounted input dataset or ephemeral cache. |
| `sha256_raw` | SHA-256 of original bytes. |
| `sha256_canonical` | SHA-256 after canonical RGB decode/resize serialization. |
| `phash` | Perceptual hash used for near-duplicate checks. |
| `modality` | `dermoscopy`, `clinical`, `dermatopathology`, `unknown`. |
| `diagnosis_label` | Source label, normalized only for audit/probe diagnostics. |
| `patient_id` | Source patient ID if available, otherwise null. |
| `lesion_id` | Source lesion ID if available, otherwise null. |
| `split` | `pretrain_train`, `pretrain_val`, `audit_only`, or `excluded`. |
| `ham10000_exclusion_reason` | Null for accepted rows, otherwise exact reason. |
| `manual_review_status` | `not_needed`, `pending`, `cleared`, or `excluded`. |

### 4.4 HAM10000 exclusion gates

The audit must produce `exp009_ham10000_exclusion_report.json` and fail if any accepted row violates these gates:

| Gate | Rule |
|---|---|
| Source metadata | Reject if source collection, ISIC ID, image ID, lesion ID, or filename maps to known HAM10000 records. |
| Exact hash | Reject if `sha256_raw` or `sha256_canonical` matches the HAM10000 reference set. |
| Perceptual hash | Reject if pHash Hamming distance to any HAM10000 image is `<= 4`. |
| Ambiguous near-duplicate | Queue distance `5-10` for manual review; exclude unless manually cleared with saved evidence. |
| Missing provenance | Reject if license or source identity is missing. |
| Split leakage | Never include HAM10000 images in pretraining train or validation splits. |

Acceptance criteria for the full pretrain:

- `accepted_images >= 50,000`, with a written source distribution table.
- `exact_ham10000_overlap_accepted == 0`.
- `high_confidence_near_duplicate_accepted == 0`.
- `manual_review_pending_accepted == 0`.
- `license_unresolved_accepted == 0`.
- At least 70% of accepted images are dermoscopy, or the run is explicitly labelled a mixed skin-image pretrain rather than a dermoscopy pretrain.

If fewer than 50,000 clean dermoscopy-equivalent images remain, do not spend full pretrain compute without revising the scientific claim. A small clean corpus would test low-data SSL, not the intended dermoscopy-domain transfer question.

---

## 5. Training design

### 5.1 Architecture and objective

Target architecture:

- Vision encoder: ViT-B/14, DINOv2-compatible output interface.
- Input: 224 x 224 RGB.
- Precision: bf16 where supported, fp16 fallback.
- Optimizer: AdamW or the objective library's DINO/MIM default, with weight decay schedule.
- Augmentation: standard SSL multi-crop or MIM augmentations, but no use of EXP-004 nuisance labels.
- Target length: 100k to 200k optimizer steps, with 200k as the paper-grade target if pilot throughput and validation curves justify it.

Objective choice:

1. Prefer a maintained implementation over hand-rolled SSL. Candidates: `lightly`, `timm` MAE-style training, or a small DINO/DINOv2-compatible training wrapper if DDP support is needed.
2. Record the final choice in the config and report. The key experimental axis is clean non-HAM10000 dermoscopy pretraining, not a novel SSL objective.
3. Keep the output embedding interface compatible with the current `embeddings.py` artifact contract.

### 5.2 Checkpoint selection

Allowed selection signals:

- Self-supervised validation loss on the non-HAM10000 validation split.
- Feature collapse checks: embedding norm stability, dimension variance floor, nearest-neighbor diversity.
- Optional frozen linear probe on non-HAM10000 labels if enough labels are available.

Disallowed selection signals:

- EXP-004 validation AUROC.
- EXP-004 test AUROC.
- Any stable/changing proxy metric on HAM10000.

The final HAM10000 EXP-004 probe is a single downstream evaluation after checkpoint selection is complete.

### 5.3 Pretrain artifacts

The pretraining job must upload:

```text
exp009-dinov2-vitb14-clean-ssl-200k-v1/
  config.yaml
  corpus_manifest_fingerprint.json
  exp009_corpus_audit.json
  exp009_ham10000_exclusion_report.json
  checkpoints/
    checkpoint_last.pt
    checkpoint_selected.pt
  metrics/
    pretrain_metrics.json
    progress.jsonl
    collapse_checks.json
    validation_probe.json
  encoder_card.md
  environment.json
```

The probe job must upload the existing run artifacts:

```text
ham10000-hf-exp009-clean-dinov2-probe-v1/
  config.yaml
  metrics.json
  baseline_metrics.json
  benchmark_report.json
  model_card.md
  embedding_index.json
  artifacts/reports/jepa_training_report.json
  artifacts/reports/baseline_failure_cases.json
  artifacts/reports/data_audit.json
```

---

## 6. Hugging Face Jobs pipeline

Use two Hub dataset repos:

| Repo | Visibility | Purpose |
|---|---|---|
| `<namespace>/derma-jepa-exp009-inputs` | private or gated | Raw/cache inputs if license permits, source manifests, HAM10000 reference hashes. |
| `<namespace>/derma-jepa-exp009-runs` | private during execution, public only after license review | Audit reports, checkpoints, metrics, probe outputs. |

### 6.1 Stage 0: local dry-run checks

Before launching hosted jobs:

```bash
uv run ruff check .
uv run pytest
HF_JOBS_DRY_RUN=1 ./scripts/hf_jobs_train_bundle.sh
```

Once the EXP-009 launchers exist, each launcher must support `HF_JOBS_DRY_RUN=1`.

### 6.2 Stage 1: corpus audit job

Planned command:

```bash
DERMA_JEPA_RUN_ID=exp009-corpus-audit-v1 \
DERMA_JEPA_EXP009_CORPUS_CONFIG=configs/pretrain/exp009_corpus.yaml \
HF_JOBS_FLAVOR=cpu-upgrade \
HF_JOBS_TIMEOUT=24h \
HF_JOBS_VOLUME="hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro" \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=exp009-corpus-audit-v1 \
  ./scripts/hf_jobs_exp009_corpus_audit.sh
```

Use `cpu-upgrade` first. Move to `cpu-xl` if image decoding and pHash generation bottleneck on CPU/RAM. Use `a10g-large` only if the audit implementation includes GPU image preprocessing.

Gate to continue: the audit report satisfies all acceptance criteria in section 4.4.

### 6.3 Stage 2: SSL smoke job

Purpose: validate dependencies, image loading, checkpoint writing, artifact upload, and non-collapse on a small step count.

```bash
DERMA_JEPA_RUN_ID=exp009-dinov2-vitb14-clean-ssl-smoke-v1 \
DERMA_JEPA_PRETRAIN_CONFIG=configs/pretrain/exp009_dinov2_vitb14_non_ham10000.yaml \
DERMA_JEPA_PRETRAIN_MAX_STEPS=1000 \
HF_JOBS_FLAVOR=a10g-large \
HF_JOBS_TIMEOUT=3h \
HF_JOBS_VOLUME="hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro" \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=exp009-dinov2-vitb14-clean-ssl-smoke-v1 \
  ./scripts/hf_jobs_exp009_pretrain.sh
```

Gate to continue:

- Job finishes within timeout.
- Loss decreases or produces a sensible SSL diagnostic trend.
- `collapse_checks.json` passes.
- Checkpoint can be loaded locally by the planned embedding exporter.

### 6.4 Stage 3: 10k-step pilot

Purpose: measure real throughput and derive cost from observed steps/hour.

```bash
DERMA_JEPA_RUN_ID=exp009-dinov2-vitb14-clean-ssl-pilot-10k-v1 \
DERMA_JEPA_PRETRAIN_MAX_STEPS=10000 \
HF_JOBS_FLAVOR=4xa100-large \
HF_JOBS_TIMEOUT=12h \
HF_JOBS_VOLUME="hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro" \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=exp009-dinov2-vitb14-clean-ssl-pilot-10k-v1 \
  ./scripts/hf_jobs_exp009_pretrain.sh
```

If DDP is not implemented yet, use `h200` or `a100-large` for the pilot instead of `4xa100-large`.

Pilot cost formula:

```text
estimated_full_hours = target_steps / observed_steps_per_hour
estimated_full_cost = estimated_full_hours * hourly_rate
```

Gate to continue:

- Projected full-pretrain cost fits the approved budget.
- Projected full-pretrain wall time fits the timeout and retry plan.
- Validation/collapse metrics are healthy.

### 6.5 Stage 4: full pretrain

Recommended launch if DDP is available:

```bash
DERMA_JEPA_RUN_ID=exp009-dinov2-vitb14-clean-ssl-200k-v1 \
DERMA_JEPA_PRETRAIN_MAX_STEPS=200000 \
HF_JOBS_FLAVOR=4xa100-large \
HF_JOBS_TIMEOUT=72h \
HF_JOBS_DETACH=1 \
HF_JOBS_VOLUME="hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro" \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=exp009-dinov2-vitb14-clean-ssl-200k-v1 \
  ./scripts/hf_jobs_exp009_pretrain.sh
```

Recommended launch if DDP is not ready:

```bash
DERMA_JEPA_RUN_ID=exp009-dinov2-vitb14-clean-ssl-200k-v1 \
DERMA_JEPA_PRETRAIN_MAX_STEPS=200000 \
HF_JOBS_FLAVOR=h200 \
HF_JOBS_TIMEOUT=96h \
HF_JOBS_DETACH=1 \
HF_JOBS_VOLUME="hf://datasets/<namespace>/derma-jepa-exp009-inputs:/mnt/exp009:ro" \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=exp009-dinov2-vitb14-clean-ssl-200k-v1 \
  ./scripts/hf_jobs_exp009_pretrain.sh
```

Do not rely on the default HF Jobs timeout. The official default is 30 minutes, which is too short for every EXP-009 stage except dry-run failures.

### 6.6 Stage 5: HAM10000 embedding export and EXP-004 probe

After selecting the pretrain checkpoint, create `configs/data/ham10000_hf_mounted_exp009.yaml` from EXP-004 with these changes only:

```yaml
embeddings:
  models:
    - model_id: exp009_dinov2_vitb14_clean_ssl
      kind: local_dinov2_checkpoint
      checkpoint_uri: hf://datasets/<namespace>/derma-jepa-exp009-runs/exp009-dinov2-vitb14-clean-ssl-200k-v1/checkpoints/checkpoint_selected.pt
      batch_size: 16
      device: auto

training:
  model_id: jepa_predictor_ham10000_exp009_clean_dinov2_v1
  embedding_model_id: exp009_dinov2_vitb14_clean_ssl
```

Launch:

```bash
DERMA_JEPA_RUN_ID=ham10000-hf-exp009-clean-dinov2-probe-v1 \
HF_JOBS_FLAVOR=a10g-large \
HF_JOBS_TIMEOUT=12h \
HF_JOBS_DETACH=1 \
HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
HF_OUTPUT_PATH=ham10000-hf-exp009-clean-dinov2-probe-v1 \
  ./scripts/hf_jobs_ham10000_exp009.sh
```

### 6.7 Stage 6: five-seed probe sweep

After the primary run completes and artifacts verify:

```bash
for seed in 1 2 3 4; do
  BASE_CONFIG=configs/data/ham10000_hf_mounted_exp009.yaml \
  SEED="$seed" \
  SWEEP_TAG=exp009-clean-dinov2 \
  HF_OUTPUT_REPO_ID="<namespace>/derma-jepa-exp009-runs" \
  HF_JOBS_DETACH=1 \
    ./scripts/hf_jobs_seed_sweep.sh
done
```

Aggregate with `scripts/aggregate_seed_sweep.py`, adding the primary seed `20260422` plus seeds 1-4. The EXP-009 report should quote seed mean, seed standard deviation, and bootstrap CI for the primary run.

---

## 7. Machine and cost estimates

Hugging Face Jobs bill per minute while the Job is `Starting` or `Running`. Jobs are not billed during build, and a failed running job is automatically suspended. Set explicit timeouts for every stage.

Prices below use the official Jobs pricing page checked on 2026-04-29. Re-check with `hf jobs hardware` immediately before launch.

| Stage | Recommended flavor | Listed price | Time estimate | Compute estimate |
|---|---|---:|---:|---:|
| Corpus audit, normal | `cpu-upgrade` | `$0.03/hr` | 6-24 hr | `< $1` |
| Corpus audit, heavy decode/pHash | `cpu-xl` | `$1.00/hr` | 4-10 hr | `$4-$10` |
| Smoke pretrain | `a10g-large` | `$1.50/hr` | 1-3 hr | `$1.50-$4.50` |
| 10k-step pilot, DDP | `4xa100-large` | `$10.00/hr` | 2-8 hr | `$20-$80` |
| 10k-step pilot, single-GPU | `h200` | `$5.00/hr` | 2-6 hr | `$10-$30` |
| Full 200k pretrain, preferred if DDP works | `4xa100-large` | `$10.00/hr` | 24-60 hr | `$240-$600` |
| Full 200k pretrain, simpler single-GPU | `h200` | `$5.00/hr` | 36-80 hr | `$180-$400` |
| Full 200k pretrain, budget fallback | `a100-large` | `$2.50/hr` | 60-140 hr | `$150-$350` |
| Full 200k pretrain, avoid unless necessary | `a10g-large` | `$1.50/hr` | 150-300 hr | `$225-$450` |
| Primary HAM10000 probe | `a10g-large` | `$1.50/hr` | 1.5-2.5 hr | `$2.25-$3.75` |
| Four extra seed probes | `a10g-large` x4 | `$1.50/hr` | 6-10 total GPU-hr | `$9-$15` |

Recommended budget envelope:

| Scenario | When it applies | Compute budget |
|---|---|---:|
| Minimal validation only | Audit + smoke + pilot + no full pretrain | `$30-$120` |
| Expected EXP-009 | Audit + smoke + pilot + one full pretrain + five probe seeds | `$350-$800` |
| Conservative reserve | One failed full pretrain retry, volume mount flakes, extra pilot | `$1,200-$1,800` |

Human time estimate:

| Work item | Estimate |
|---|---:|
| Source/license triage and input dataset assembly | 1-3 working days if data access is straightforward; 1-2 weeks if source terms or downloads are messy. |
| Corpus audit implementation and report schema | 2-3 working days. |
| SSL pretraining command and checkpoint export | 3-5 working days single-GPU; add 2-4 days for DDP/multinode hardening. |
| HF launchers and dry-run validation | 0.5-1 working day. |
| Smoke/pilot/full hosted execution | 2-5 calendar days, mostly waiting, if jobs are healthy. |
| Probe seed sweep and EXP-009 report | 1 working day. |

End-to-end: 7-12 focused working days in the optimistic path; 2-3 weeks if data provenance or DDP becomes the critical path.

---

## 8. Stop/go gates

| Gate | Stop condition | Continue condition |
|---|---|---|
| G0: licenses | Any accepted source lacks redistribution/use permission for this research run. | Every accepted row has source and license metadata. |
| G1: corpus | Fewer than 50k clean images, unresolved manual-review rows, or any accepted HAM10000 exact/high-confidence near duplicate. | Clean manifest passes section 4.4. |
| G2: smoke | Dependency failure, checkpoint cannot load, or collapse checks fail. | 1k-step job uploads loadable checkpoint and non-collapsed embeddings. |
| G3: pilot | Throughput projects beyond budget, or validation metrics are flat/collapsed. | 10k-step pilot supports budget and training-health assumptions. |
| G4: full pretrain | Full run fails or selected checkpoint is collapsed. | Selected checkpoint passes non-HAM10000 validation and collapse checks. |
| G5: probe | Probe artifacts incomplete or config differs from EXP-004 beyond encoder identity. | Primary probe produces the standard metric/report bundle. |
| G6: seed sweep | Seed variance crosses decision-band boundaries. | Quote seed mean/std and classify only if the band remains stable. |

If G4 fails, do not interpret a low EXP-009 AUROC as evidence for HAM10000 overlap. The conclusion would be "clean pretraining attempt failed," not "clean dermoscopy transfer failed."

---

## 9. Reporting requirements

The final EXP-009 report should include:

1. Corpus source distribution table by `source_dataset` and `modality`.
2. HAM10000 exclusion report with exact-hash count, pHash-near count, manual review count, and excluded-row counts.
3. Pretraining hardware, wall time, objective, target steps, achieved steps, and cost.
4. Pretrain-health metrics and checkpoint-selection rule.
5. EXP-004 probe config diff proving the only changed axis is the encoder.
6. Primary AUROC with bootstrap CI.
7. Five-seed mean/std and range.
8. Decision-table classification and the exact claim revision implied by that band.
9. A negative-results section if the pretrain fails or lands in the low band.

Suggested report path after completion:

```text
docs/experiments/EXP-009-non-ham10000-dermoscopy-pretrain-v1.md
```

Until then, this file is the launch contract and should be updated only when a design decision changes before the first hosted job.
