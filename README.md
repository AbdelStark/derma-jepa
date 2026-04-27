# DermaJEPA

> JEPA-style latent trajectory probe for skin lesion monitoring on HAM10000.

DermaJEPA tests whether a JEPA-style representation, learned over a frozen vision backbone, can separate stable lesions under nuisance variation from genuinely changing lesions on a leakage-controlled longitudinal proxy. Lesion monitoring is fundamentally a change-detection problem; the working hypothesis is that the useful signal lives in latent space rather than in raw pixel differences.

## Working hypothesis

A JEPA-style predictor can learn a lesion representation in which stable lesions remain close under nuisance variation and genuine lesion change appears as a structured latent departure.

## Problem statement

Repeated lesion photographs vary heavily because of illumination, angle, skin texture, hair occlusion, camera quality, zoom, and framing. A clinically useful monitoring system has to separate nuisance variation from lesion evolution. Existing pipelines either optimise for one-shot diagnosis or rely on handcrafted image-difference heuristics that are brittle under real-world smartphone capture. This repository runs a controlled set of experiments on HAM10000 to test whether JEPA-style probes over modern frozen backbones do better.

## Project status

Nine primary-tier experiments have run on Hugging Face Jobs against `abdelstark/ham10000`, using a third-family-held-out nuisance protocol to test out-of-distribution generalisation of the predictor. Results are summarised in `docs/experiments/`, with each report self-contained for citation.

| Configuration on the EXP-004 nuisance-held-out proxy | Test AUROC on `strong_held_out_2` |
|---|---:|
| Pixel L2 (cheap baseline) | 0.580 |
| Frozen DINOv2 ViT-B/14 + linear predictor (EXP-004) | 0.249 |
| Frozen DINOv2 ViT-B/14 + Adam-tuned MLP (EXP-006a) | 0.248 |
| Frozen OpenAI CLIP ViT-B/16 + linear predictor (EXP-006b) | 0.286 |
| Frozen BiomedCLIP ViT-B/16 + linear predictor (EXP-008, 5-seed mean) | 0.329 ± 0.012 |
| **Frozen DermLIP ViT-B/16 + linear predictor (EXP-007, 5-seed mean)** | **0.944 ± 0.003** |

Frozen natural-image and frozen general-medical backbones produce below-random test AUROC on the third unseen nuisance family across two backbones, three scaffolds, and two optimisers. A frozen dermoscopy-specific backbone (DermLIP, CLIP-trained on Derm1M) lifts test AUROC to 0.944 ± 0.003 (5 seeds). DermLIP's pretraining corpus almost certainly includes HAM10000, so the contribution of dermoscopy-domain transfer versus HAM10000 image-level overlap is not yet partitioned. EXP-009 (a DINOv2 self-pretrained on a non-HAM10000 dermoscopy corpus) addresses this.

See `docs/experiments/README.md` for the full index, including each report's headline number, baseline, delta, and outcome.

## Scope of v1

In scope:

- Public-data audit and longitudinal-proxy construction on HAM10000.
- Linear and small MLP JEPA-style predictors over frozen vision backbones.
- Pixel L2, SSIM, and frozen-embedding-cosine baselines.
- Bootstrap CI evaluation, leakage probes, and fixed-TPR / EER reporting.
- Reproducible Hugging Face Jobs launchers + run-archive convention.

Out of scope:

- Diagnostic or treatment recommendations.
- Production medical-device claims.
- JEPA pretraining of the backbone from scratch.
- Real longitudinal data; HAM10000 is cross-sectional and the proxy is synthetic throughout.

## Data sources

- HAM10000 (primary; uploaded as `abdelstark/ham10000` on the Hub for hosted runs).
- ISIC Archive components excluding HAM10000 (planned for EXP-009 SSL pretrain corpus).
- PAD-UFES-20 (kept as a fallback / out-of-distribution probe; not currently used in the EXP-004 proxy).

## Repository structure

- `src/derma_jepa/` — package, CLI, manifest contracts, fixture pipeline, baselines, hf-run helpers, and demo export.
- `configs/manifest/fixture.yaml` — deterministic fixture-tier pipeline config.
- `configs/data/` — HAM10000 audit and per-experiment proxy configs.
- `data/README.md` — local data layout, source/citation notes, and leakage rules.
- `tests/` — contract, metric, and end-to-end fixture pipeline tests.
- `docs/prd/PRD.md` — product requirements.
- `docs/spec/SYSTEM-SPEC.md` — system architecture and design contracts.
- `docs/spec/RESEARCH.md` — research notes and open questions.
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan.
- `docs/spec/MVP-STATUS.md` — current implementation state.
- `docs/rfcs/` — decision records that lock the design before code expands.
- `docs/experiments/` — per-run reports (one Markdown file per primary-tier run) plus the seed-sweep summary.
- `docs/runbooks/` — operational playbooks (Hugging Face Jobs, the HAM10000 JEPA loop).

## Fixture pipeline

The first implementation milestone is a contract-first fixture pipeline. It uses deterministic synthetic images to prove the repository mechanics before public data, DINOv2 embeddings, or JEPA-style predictor training.

Install the development environment:

```bash
uv sync --extra dev
```

Run the full fixture contract:

```bash
uv run derma-jepa fixture pipeline --config configs/manifest/fixture.yaml
```

This builds and validates the synthetic pair manifest, exports deterministic fixture embeddings, evaluates pixel L2, SSIM, and embedding-distance baselines, writes a self-contained run directory, validates the fixture acceptance gate, and exports a local demo bundle.

Generated outputs:

- `runs/fixture-contract-v1/` — manifests, metrics, baseline report, model card, logs, embeddings, plot, benchmark report, and demo case JSON.
- `artifacts/demo/fixture-contract-v1/` — portable fixture demo bundle with `demo_case.json`, copied synthetic images, and `index.html`.

Open the exported demo entrypoint:

```bash
uv run derma-jepa demo --artifact artifacts/demo/fixture-contract-v1
```

## Public-data audit path

The repository does not vendor public images. Place raw HAM10000 downloads under `data/raw/ham10000/` as documented in `data/README.md`, then run:

```bash
uv run derma-jepa data audit --config configs/data/ham10000.yaml
uv run derma-jepa manifest build --config configs/data/ham10000.yaml
uv run derma-jepa embed --config configs/data/ham10000.yaml
uv run derma-jepa baseline eval --config configs/data/ham10000.yaml
```

The public-data manifest builder writes normalised metadata, an audit report, lesion-ID-aware train/val/test manifests, post-split stable nuisance variants, and a manual gold-audit subset. The embedding command exports the configured DINOv2 ViT-S/14 and ViT-B/14 image embeddings when optional model dependencies are installed:

```bash
uv sync --extra model
```

Baseline evaluation then reports pixel L2, SSIM, and embedding-distance metrics plus failure-case templates for manual review. This remains a longitudinal-proxy research path, not a diagnostic workflow.

Validate the codebase:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

## Hosted compute via Hugging Face Jobs

The training scaffold runs on Hugging Face Jobs against the dataset-mount at `/data`. See `docs/runbooks/huggingface-jobs.md` for the authoritative walkthrough.

```bash
# uv-run launcher for a public GitHub ref
./scripts/hf_jobs_train.sh

# private-wheel bundle launcher (no GitHub access required from the Job)
./scripts/hf_jobs_train_bundle.sh

# first real public-data run with the dataset mounted at /data
./scripts/hf_jobs_ham10000_primary.sh
# see docs/runbooks/ham10000-jepa-playbook.md for the operational playbook

# per-experiment launchers (one per EXP-XXX config)
./scripts/hf_jobs_ham10000_exp006a.sh   # Adam MLP on DINOv2
./scripts/hf_jobs_ham10000_exp006b.sh   # OpenAI CLIP backbone swap
./scripts/hf_jobs_ham10000_exp007.sh    # DermLIP backbone swap
./scripts/hf_jobs_ham10000_exp008.sh    # BiomedCLIP backbone swap

# seed-sweep launcher (parametric over a base config + seed)
BASE_CONFIG=configs/data/ham10000_hf_mounted_exp007.yaml \
SEED=1 SWEEP_TAG=dermlip-exp007 \
  ./scripts/hf_jobs_seed_sweep.sh

# fetch and summarise a completed run uploaded under run_id
uv run derma-jepa hf-run summary \
  --repo-id "$HF_USER/derma-jepa-runs" \
  --run-id <run_id>
```

Both launchers pin hosted dependency versions from `scripts/hf_jobs_constraints.txt` so numpy, scipy, torch, transformers, and open-clip-torch do not float between Jobs. Keep the pin file aligned with `uv.lock`.

## Command surface

```bash
derma-jepa data audit --config configs/manifest/fixture.yaml
derma-jepa manifest build --config configs/manifest/fixture.yaml
derma-jepa manifest build --config configs/data/ham10000.yaml
derma-jepa embed --config configs/manifest/fixture.yaml
derma-jepa embed --config configs/data/ham10000.yaml
derma-jepa baseline eval --config configs/manifest/fixture.yaml
derma-jepa baseline eval --config configs/data/ham10000.yaml
derma-jepa eval --config configs/manifest/fixture.yaml
derma-jepa benchmark --run runs/fixture-contract-v1
derma-jepa demo export --run runs/fixture-contract-v1 --out artifacts/demo/fixture-contract-v1
derma-jepa demo --artifact artifacts/demo/fixture-contract-v1
derma-jepa train --config configs/train/jepa_predictor.yaml
derma-jepa hf-run summary --repo-id <hub-repo> --run-id <run_id>
./scripts/hf_jobs_train.sh
./scripts/hf_jobs_train_bundle.sh
./scripts/hf_jobs_seed_sweep.sh
```

## Build principle

The repository is spec-led: every primary-tier run is preceded by an RFC or an experiment plan, and every result is captured in a self-contained report under `docs/experiments/`. The fixture pipeline is the first runnable proof that the manifest, preprocessing, baseline, metric, artifact, and demo contracts can work end to end.
