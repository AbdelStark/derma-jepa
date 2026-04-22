# DermaJEPA

> JEPA-based latent trajectory modeling for skin lesion monitoring

DermaJEPA is a research project exploring whether JEPA-style representation learning can make lesion monitoring more robust than pixel-space or diagnosis-first approaches. The central idea is simple: skin monitoring is primarily a change-detection problem over time, and the useful signal should live in latent space rather than in raw pixel differences.

## Core thesis

A JEPA-style encoder can learn a lesion representation where stable lesions remain close under nuisance variation and meaningful lesion evolution appears as a structured latent trajectory departure.

## Problem statement

Repeated lesion photographs vary heavily because of illumination, angle, skin texture, hair occlusion, camera quality, zoom, and framing. A clinically useful monitoring system has to separate nuisance variation from lesion evolution. Most existing pipelines either optimize for one-shot diagnosis or rely on handcrafted image-difference heuristics that are too brittle for real-world smartphone capture.

## What v1 is

- Fine-tune a compact image JEPA encoder on public dermatology corpora.
- Construct a pseudo-longitudinal evaluation setup using same-lesion or lesion-group trajectories plus nuisance-heavy augmentations.
- Implement explicit latent drift metrics and a monitoring score instead of a diagnosis classifier.
- Ship a local monitoring demo with timeline review, trajectory visualization, and stable-vs-changing examples.

## What v1 is not

- No diagnostic or treatment recommendation engine.
- No production medical-device claim.
- No full JEPA pretraining from scratch.
- No cloud-first pipeline requirement for the demo.

## Candidate data sources

- ISIC Archive
- HAM10000
- PAD-UFES-20
- Optional same-lesion subset curation if enough repeated examples exist

## Success criteria

- Latent drift separates stable-vs-changing cases better than pixel-difference and generic frozen-embedding baselines.
- The representation is robust to lighting, angle, crop, and phone-quality perturbations.
- The demo produces an immediately legible monitoring narrative in under three minutes.

## Milestones

1. Dataset audit and longitudinal-proxy design
2. Baseline embedding and nuisance-robustness checks
3. Trajectory scoring implementation
4. Monitoring UI / demo flow
5. Evaluation pass and artifact export

## Repository structure

- `src/derma_jepa/` — package, CLI, manifest contracts, fixture pipeline, baselines, and demo export
- `configs/manifest/fixture.yaml` — deterministic fixture-tier pipeline config
- `configs/data/ham10000.yaml` — HAM10000-compatible public-data audit and proxy-manifest config
- `data/README.md` — local data layout, source/citation notes, and leakage rules
- `tests/` — contract, metric, and end-to-end fixture pipeline tests
- `docs/prd/PRD.md` — product requirements
- `docs/spec/SYSTEM-SPEC.md` — system architecture and design contracts
- `docs/spec/RESEARCH.md` — research notes and open questions
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan
- `docs/spec/MVP-STATUS.md` — current implementation state
- `docs/rfcs/` — decision records that lock the design before code expands

## Fixture pipeline

The first implementation milestone is a contract-first fixture pipeline. It uses
deterministic synthetic images to prove the repository mechanics before public
data, DINOv2 embeddings, or JEPA-style predictor training.

Install the development environment:

```bash
uv sync --extra dev
```

Run the full fixture contract:

```bash
uv run derma-jepa fixture pipeline --config configs/manifest/fixture.yaml
```

This command builds and validates the synthetic pair manifest, exports
deterministic fixture embeddings, evaluates pixel L2, SSIM, and embedding-distance
baselines, writes a self-contained run directory, validates the fixture acceptance
gate, and exports a local demo bundle.

Generated outputs:

- `runs/fixture-contract-v1/` — manifests, metrics, baseline report, model card,
  logs, embeddings, plot, benchmark report, and demo case JSON
- `artifacts/demo/fixture-contract-v1/` — portable fixture demo bundle with
  `demo_case.json`, copied synthetic images, and `index.html`

Open the exported demo entrypoint:

```bash
uv run derma-jepa demo --artifact artifacts/demo/fixture-contract-v1
```

## Public data audit path

Milestone 2 now has a HAM10000-compatible local data path. The repo still does
not vendor public images. Place raw downloads under `data/raw/ham10000/` as
documented in `data/README.md`, then run:

```bash
uv run derma-jepa data audit --config configs/data/ham10000.yaml
uv run derma-jepa manifest build --config configs/data/ham10000.yaml
uv run derma-jepa embed --config configs/data/ham10000.yaml
uv run derma-jepa baseline eval --config configs/data/ham10000.yaml
```

The public-data manifest builder writes normalized metadata, an audit report,
patient/lesion-aware train/validation/test manifests, post-split stable nuisance
variants, and a manual gold-audit subset. The embedding command exports the
configured DINOv2 ViT-S/14 and ViT-B/14 image embeddings when optional model
dependencies are installed:

```bash
uv sync --extra model
```

Baseline evaluation then reports pixel L2, SSIM, and embedding-distance metrics,
plus failure-case templates for manual review. This is still a longitudinal-proxy
research path, not a diagnostic workflow.

Validate the codebase:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

## Command surface

The MVP command surface is locked even though later model/data commands remain
gated by milestone:

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
./scripts/hf_jobs_train_bundle.sh
```

## Build principle

The repository remains spec-led, but implementation has started. The fixture
pipeline is the first runnable proof that the manifest, preprocessing, baseline,
metric, artifact, and demo contracts can work end to end.
