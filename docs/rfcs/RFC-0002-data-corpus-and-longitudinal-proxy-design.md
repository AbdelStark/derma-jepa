# RFC-0002 - data corpus and longitudinal-proxy design

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 4, 5, 6, 7, and 8.

## Decision

The MVP uses a longitudinal-proxy task because public dermatology datasets do
not provide enough clean, true longitudinal progression signal for the initial
claim.

Stable pairs:

- same source lesion image under nuisance perturbations
- same-lesion duplicate images when lesion identifiers make this reliable

Changing pairs:

- different lesion images matched to reduce trivial shortcuts
- matching priority:
  - same patient when available
  - same anatomical site when available
  - same diagnosis class when available
  - visually similar negatives from nearest-neighbor mining

Changing pairs are proxy departures from lesion identity or morphology. They
must not be presented as biological progression.

## Dataset priority

Primary:

- ISIC/HAM10000-style dermoscopic images for scale and research familiarity

External stress:

- PAD-UFES-20 for smartphone/clinical-image domain-shift reporting

Optional extension:

- selected ISIC Archive subsets after metadata and leakage audits

## Scale

- fixture tier: 20-100 synthetic or license-safe images
- gold audit tier: 100-300 manually inspected public images/pairs
- primary MVP tier: HAM10000-scale, roughly 10k dermoscopic images
- extended ISIC tier: 25k-50k selected images after the primary path works
- external stress tier: full PAD-UFES-20, around 2.3k images

## Leakage policy

- Split by lesion ID when lesion IDs exist.
- Split by patient ID when patient IDs exist.
- Generate stable augmentations after splitting, never before.
- Use source-aware reporting for mixed archives/devices/sites.
- Mark any result that cannot satisfy leakage constraints as exploratory.

## Consequences

The first credible result depends on manifest quality, not model size. Data
audit and leakage-risk notes must exist before full training.

## Acceptance condition

This RFC is satisfied when a generated manifest records image IDs, pair labels,
split, source dataset, metadata, preprocessing profile, augmentation recipe, and
pair-construction reason codes.
