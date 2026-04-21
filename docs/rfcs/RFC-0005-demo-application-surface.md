# RFC-0005 - demo application surface

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 12, 13, 14, 15, and 17.

## Decision

The MVP demo is a local MacBook Pro dashboard over deterministic exported cases.
It is not a phone app and not an open-ended medical app.

Required views:

- case timeline with source metadata
- latent drift chart with baseline comparison
- embedding-space view or nearest-neighbor panel
- nuisance stress view
- failure-case view
- run provenance panel

The demo must run from exported artifacts and must not require live training or
hidden raw-data cleanup. Optional on-demand inference for one uploaded pair can
be added only after the deterministic path works.

## Runtime contract

MacBook Pro tier:

- local dashboard
- precomputed artifact viewing
- small eval/report slices
- optional lightweight inference

GB10 tier:

- full embedding export
- JEPA-style predictor/projection training
- full benchmark and eval jobs
- demo artifact generation for transfer back to the MacBook Pro

## Consequences

Demo quality depends on provenance, baseline comparison, and visible failure
cases. UI polish cannot substitute for benchmark artifacts.

## Acceptance condition

This RFC is satisfied when `derma-jepa demo --artifact artifacts/demo/<run_id>`
opens the local dashboard from exported artifacts and shows the required views.
