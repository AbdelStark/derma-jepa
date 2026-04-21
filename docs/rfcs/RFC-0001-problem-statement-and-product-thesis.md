# RFC-0001 - problem statement and product thesis

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 1, 2, 17, 19, and 20.

## Decision

DermaJEPA is a monitoring-first research project, not a diagnosis product. The
MVP evaluates whether a JEPA-style latent trajectory model can separate
nuisance-induced visual drift from meaningful lesion-change proxies better than
simple pixel/SSIM baselines and generic frozen vision embeddings on public
dermatology data.

The MVP may describe itself as a restricted latent world-model prototype only in
the narrow sense that it learns a predictive model over image-derived lesion
representations. It must not claim to model disease progression, melanoma
evolution, cancer risk, or patient state.

## Non-claims

The MVP must not claim:

- diagnostic accuracy
- clinical validity
- melanoma detection
- treatment recommendation
- real patient monitoring
- medical-device readiness
- full JEPA pretraining from scratch
- a general dermatology foundation model

## Target audience

Primary:

- academic ML scientists
- ML researchers
- world-model researchers

Secondary:

- future contributors who need a reproducible research engineering surface

## Consequences

Implementation must optimize for empirical honesty before demo polish. A
negative or inconclusive result is still valid if the task, baselines, leakage
policy, and artifacts are complete.

## Acceptance condition

This RFC is satisfied when the README, demo, reports, and model cards all use
monitoring/proxy-task language and avoid diagnostic or clinical-product claims.
