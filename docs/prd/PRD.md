# DermaJEPA — PRD

## 1. Product thesis

A JEPA-style encoder can learn a lesion representation where stable lesions remain close under nuisance variation and meaningful lesion evolution appears as a structured latent trajectory departure.

## 2. Product goal

Build a compact, technically credible v1 that demonstrates the thesis with a working demo and evaluation surface, without requiring large-lab training budgets or broad platform scope.

## 3. Problem

Repeated lesion photographs vary heavily because of illumination, angle, skin texture, hair occlusion, camera quality, zoom, and framing. A clinically useful monitoring system has to separate nuisance variation from lesion evolution. Most existing pipelines either optimize for one-shot diagnosis or rely on handcrafted image-difference heuristics that are too brittle for real-world smartphone capture.

## 4. v1 scope

- Fine-tune a compact image JEPA encoder on public dermatology corpora.
- Construct a pseudo-longitudinal evaluation setup using same-lesion or lesion-group trajectories plus nuisance-heavy augmentations.
- Implement explicit latent drift metrics and a monitoring score instead of a diagnosis classifier.
- Ship a local monitoring demo with timeline review, trajectory visualization, and stable-vs-changing examples.

## 5. Explicit non-goals

- No diagnostic or treatment recommendation engine.
- No production medical-device claim.
- No full JEPA pretraining from scratch.
- No cloud-first pipeline requirement for the demo.

## 6. Users

Primary users:
- researchers and engineers evaluating whether the JEPA framing actually improves the target problem
- developers/operators who need a concrete, inspectable demo rather than a vague claim

Secondary users:
- open-source contributors joining after the initial prototype
- technical readers who want the core design decisions spelled out before implementation grows

## 7. Demo requirement

The project must support a short demo that makes the thesis obvious without hidden manual setup or cloud-only dependencies.

## 8. Success criteria

- Latent drift separates stable-vs-changing cases better than pixel-difference and generic frozen-embedding baselines.
- The representation is robust to lighting, angle, crop, and phone-quality perturbations.
- The demo produces an immediately legible monitoring narrative in under three minutes.

## 9. Main risks

- weak task definition creates a fake win
- model scope expands faster than the data/eval contract becomes clear
- demo polish arrives before the underlying claim is actually supported
- baselines are too weak and make the result look better than it is

## 10. Deliverables for MVP

- reproducible data/task contract
- baseline system
- JEPA-based representation/training path
- downstream scoring or decode path
- demo surface
- evaluation report and exported artifacts
