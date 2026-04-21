# DermaJEPA MVP schedule

Status: Accepted for MVP execution
Canonical contract: `docs/spec/MVP-SPEC.md`

This schedule is milestone-based rather than calendar-based. Each milestone
closes with one explicit output artifact, one validation step, and one written
note about what changed in project understanding.

## Milestone 1 - contract-first fixture pipeline

Output artifact:

- fixture-tier run directory with config, manifests, metrics, baseline metrics,
  initial plots, logs, environment file, and one demo case JSON

Validation:

- CI runs the fixture pipeline end to end

Understanding note:

- document whether the schemas, command surface, and artifact contract were
  sufficient before real data was introduced

## Milestone 2 - data audit and baselines

Output artifact:

- data audit report plus leakage-controlled HAM10000/ISIC proxy manifest and
  baseline report

Validation:

- manifest regeneration succeeds from config and baseline metrics include
  confidence intervals

Understanding note:

- document available metadata, leakage risks, duplicate risks, and whether the
  proxy task remains defensible

## Milestone 3 - JEPA-style training

Output artifact:

- full primary-tier JEPA-style predictor run directory

Validation:

- tiny overfit/debug run succeeds, full run completes within the GB10 budget, and
  collapse checks are reported

Understanding note:

- document whether latent prediction training produced usable, non-collapsed
  representations

## Milestone 4 - evals and benchmarks

Output artifact:

- benchmark report with primary metrics, robustness, representation health,
  ablations, runtime, and qualitative cases

Validation:

- strongest baseline comparison is complete and result wording matches the
  measured evidence

Understanding note:

- document whether the thesis is supported, inconclusive, or contradicted on the
  locked proxy task

## Milestone 5 - local demo

Output artifact:

- exported demo bundle and local dashboard

Validation:

- dashboard runs on MacBook Pro from exported artifacts without raw data or live
  training

Understanding note:

- document which visualizations make the evidence legible and which failure
  cases must remain visible

## Milestone 6 - MVP closeout

Output artifact:

- model card, results report, README reproduction path, and go/no-go summary

Validation:

- definition of done in `docs/spec/MVP-SPEC.md` is checked item by item

Understanding note:

- document the next scientifically honest step: better proxy, stronger
  backbone, true longitudinal data, or different objective

## Completion rule

The MVP is not complete if any of these are missing:

- leakage-controlled manifest
- mandatory baselines
- full run directory
- confidence intervals
- failure cases
- local artifact-backed demo
- safety-language audit
- CI fixture pipeline
