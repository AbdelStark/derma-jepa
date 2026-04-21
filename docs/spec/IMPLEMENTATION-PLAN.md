# DermaJEPA implementation plan

Status: Accepted for MVP execution
Canonical contract: `docs/spec/MVP-SPEC.md`

## Goal

Build the smallest credible MVP that evaluates this thesis:

> A JEPA-style latent trajectory model can separate nuisance-induced visual
> drift from meaningful lesion-change proxies better than simple pixel/SSIM
> baselines and generic frozen vision embeddings on public dermatology data.

The implementation must prove the operational path before spending serious
training compute.

## Operating rules

- Build from `docs/spec/MVP-SPEC.md`.
- Treat the RFCs as accepted decision records, not open brainstorming docs.
- Do not implement clinical or diagnostic language.
- Do not rely on notebook-only steps for MVP results.
- Do not run full training before the fixture pipeline and baseline contracts
  work.
- Do not mark implementation milestones complete without runnable or inspectable
  artifacts.
- Preserve negative results if the JEPA-style model fails to beat the strongest
  baseline.

## Milestone 1 - contract-first fixture pipeline

Status: Complete in the fixture tier.

Purpose: prove the repository mechanics before real data/model complexity.

Deliverables:

- `pyproject.toml` and package scaffold
- Typer-based `derma-jepa` CLI skeleton
- locked command surface:
  - `derma-jepa data audit`
  - `derma-jepa manifest build`
  - `derma-jepa embed`
  - `derma-jepa baseline eval`
  - `derma-jepa train`
  - `derma-jepa eval`
  - `derma-jepa benchmark`
  - `derma-jepa demo export`
  - `derma-jepa demo`
- manifest schema and validation
- synthetic/tiny fixture dataset
- preprocessing profile implementation
- pixel/SSIM baseline on fixtures
- AUROC and bootstrap CI metric wrapper
- run directory writer
- demo export JSON for one fixture case
- CI test that runs the fixture pipeline end to end

Acceptance:

- one command or script runs fixture manifest build through eval artifact export
- generated run directory satisfies the fixture-tier artifact contract
- tests pass locally and in CI

Implemented command:

```bash
uv run derma-jepa fixture pipeline --config configs/manifest/fixture.yaml
```

## Milestone 2 - public data audit and baseline path

Purpose: establish the leakage-controlled benchmark before JEPA training.

Deliverables:

- `data/README.md` with source, license/access, expected files, checksums where
  possible, and citations
- HAM10000/ISIC indexing or download instructions
- normalized metadata tables
- leakage audit with patient/lesion/source/duplicate checks
- stable/changing proxy manifest
- gold audit subset
- pixel/SSIM baseline report
- DINOv2 ViT-S/14 and ViT-B/14 embedding export
- DINOv2 baseline report
- initial failure-case templates

Acceptance:

- primary-tier manifest can be regenerated from config
- baseline metrics include bootstrap confidence intervals
- leakage-risk note is written before model training

## Milestone 3 - JEPA-style predictor training

Purpose: test the project thesis with a compact latent prediction objective.

Deliverables:

- context/target latent dataset
- predictor/projection model
- training loop
- collapse checks
- tiny overfit/debug run
- smoke training run under 30 minutes
- full primary-tier training run targeting under 12 hours on GB10, hard cap 24
  hours
- complete run directory

Acceptance:

- model can overfit a tiny fixture slice
- full run exports metrics, embeddings, plots, logs, config, model card, and demo
  cases
- changing pairs are not trained to collapse together

## Milestone 4 - evaluation and benchmark suite

Purpose: make the result scientifically interpretable.

Deliverables:

- primary AUROC with bootstrap confidence interval
- AUPRC, equal-error-rate threshold, FPR at fixed TPR
- nuisance robustness report by family and severity
- representation health report
- ablations:
  - no JEPA predictor
  - frozen projection versus lightly adapted projection
  - masking variants
  - pair-construction variants
- runtime benchmark report
- qualitative case-study report

Acceptance:

- JEPA-style score is compared against the strongest baseline
- positive, negative, or inconclusive result wording follows the MVP spec
- no chart depends on hand-picked cases outside the manifest

## Milestone 5 - local MacBook Pro demo

Purpose: show the thesis and evidence from exported artifacts.

Deliverables:

- exported demo artifact bundle
- Streamlit or lightweight local dashboard
- case timeline view
- latent drift chart with baseline comparison
- embedding-space or nearest-neighbor view
- nuisance stress view
- failure-case view
- run provenance panel

Acceptance:

- `derma-jepa demo --artifact artifacts/demo/<run_id>` runs locally
- demo does not require raw data when exported artifacts exist
- demo copy passes safety-language audit

## Milestone 6 - MVP report and hardening

Purpose: package the work as a credible research demo.

Deliverables:

- model card
- results report
- README reproduction path
- safety/privacy/clinical-boundary note
- CI fixture gate
- go/no-go summary
- optional deterministic screen recording
- optional GB10 setup note

Acceptance:

- all MVP definition-of-done items in `docs/spec/MVP-SPEC.md` are satisfied
- remaining risks are explicit
- the project can be handed to a skeptical ML reader without hidden steps

## First vertical slice

The first implementation slice is:

```text
fixture images -> manifest -> baseline embeddings -> latent drift scores ->
AUROC report -> exported demo case -> local dashboard
```

Explicitly out of the first slice:

- GB10 training
- full dataset download
- I-JEPA checkpoint adaptation
- polished UI
- clinical wording

This slice is complete only when CI runs it end to end.
