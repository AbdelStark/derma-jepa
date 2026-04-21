# RFC-0004 - trajectory scoring and alerting

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 5, 10, 11, 14, and 15.

## Decision

The MVP exposes latent drift scores, not clinical alerts.

The scoring path is:

1. export embeddings for each image or generated pair/window
2. compute latent drift for pair/window comparisons
3. compare drift distributions for stable and changing proxy labels
4. report threshold behavior only as a research metric
5. display thresholds in the demo as references, not medical advice

Primary score:

- latent drift score per pair/window

Primary metric:

- held-out pairwise proxy change-detection AUROC with bootstrap confidence
  interval

Secondary threshold metrics:

- AUPRC
- equal-error-rate threshold
- FPR at fixed TPR
- calibration/error curves

## No clinical alert semantics

The system must not say "high risk", "consult a doctor", "melanoma likely", or
any equivalent clinical alert. Demo language may say "higher latent drift than
baseline/reference examples" and must keep the proxy-task context visible.

## Consequences

Trajectory scoring remains interpretable for researchers without making a
medical-device claim. Any threshold selected for visualization must be tied to
evaluation artifacts.

## Acceptance condition

This RFC is satisfied when drift scores, baseline scores, thresholds, and plots
are generated from run artifacts and the demo presents them as proxy-task
evidence only.
