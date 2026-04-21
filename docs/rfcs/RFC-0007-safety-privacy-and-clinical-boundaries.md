# RFC-0007 - safety, privacy, and clinical boundaries

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 1, 4, 15, 17, and 19.

## Decision

DermaJEPA MVP is a research demo over public/research-allowed dermatology data.
It is not diagnostic, not patient-facing, and not validated for medical use.

Required language:

- research demo
- longitudinal-proxy task
- latent drift
- not diagnostic
- not medical advice
- not validated for patient use

Forbidden language:

- detects melanoma
- predicts cancer
- clinical decision support
- diagnosis
- treatment recommendation
- safe for patient monitoring
- medical-device ready

## Data policy

- Do not vendor raw public datasets into the repo.
- Keep raw data under gitignored local paths.
- Commit only synthetic/license-safe fixtures.
- Respect source dataset license and citation terms.
- Do not use personal photos or patient data in MVP artifacts.
- Demo cases must carry source attribution in provenance.

## Consequences

Safety and privacy boundaries are part of the definition of done. A demo or
report with clinical overclaiming is incomplete even if the metrics pass.

## Acceptance condition

This RFC is satisfied when README, reports, model cards, run artifacts, and demo
copy pass a safety-language audit and use only public/research-allowed data.
