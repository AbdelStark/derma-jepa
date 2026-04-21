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

- `docs/prd/PRD.md` — product requirements
- `docs/spec/SYSTEM-SPEC.md` — system architecture and design contracts
- `docs/spec/RESEARCH.md` — research notes and open questions
- `docs/spec/IMPLEMENTATION-PLAN.md` — phased execution plan
- `docs/spec/MVP-STATUS.md` — current implementation state
- `docs/rfcs/` — decision records that lock the design before code expands

## Build principle

The repository is spec-first by design.
Implementation should follow only after the PRD, system spec, and RFC stack are coherent enough to make the first build phase mechanical.
