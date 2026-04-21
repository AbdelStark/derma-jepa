# RFC-0006 - evaluation and baselines

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 9, 11, 14, 19, and 20.

## Decision

No JEPA-style result is credible until mandatory baselines and leakage probes are
complete.

Mandatory baselines:

- resized/cropped pixel L2
- SSIM
- LPIPS if dependency cost is acceptable
- frozen DINOv2 ViT-S/14 cosine distance on class token and average patch token
- frozen DINOv2 ViT-B/14 cosine distance on class token and average patch token
- dermatology-supervised embedding baseline if labels are clean enough
- trivial metadata/leakage probes where metadata exists

Primary metric:

- pairwise proxy change-detection AUROC on held-out lesion/patient-aware splits

Minimum positive target:

- at least `+0.05 AUROC` over the strongest cheap baseline on the primary
  held-out split, with bootstrap confidence interval

Secondary metrics:

- AUPRC
- equal-error-rate threshold
- FPR at fixed TPR
- calibration/error curves

Benchmark suite:

- nuisance robustness by augmentation family and severity
- representation health checks
- ablations
- runtime benchmarks
- qualitative case studies

## Failure policy

If the JEPA-style model does not beat the strongest baseline, the MVP still
ships as a negative or inconclusive research artifact. Baseline removal,
threshold tuning, or cherry-picked cases are not allowed to rescue the story.

## Consequences

The report must separate metric results, robustness results, runtime results,
and qualitative cases. A polished demo without benchmark artifacts is not an
MVP.

## Acceptance condition

This RFC is satisfied when a full run directory contains complete JEPA metrics,
baseline metrics, confidence intervals, plots, failure cases, and model-card
limitations.
