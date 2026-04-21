# RFC-0003 - model stack and JEPA objective

## Status

Accepted

## Canonical spec

See `docs/spec/MVP-SPEC.md`, sections 9, 10, 11, and 12.

## Decision

The MVP implements a JEPA-style adaptation path, not full JEPA pretraining from
scratch.

The model path is:

- start from a strong frozen or lightly adapted vision foundation encoder
- train a compact projection/predictor stack over dermatology pairs/windows
- export embeddings and drift scores for downstream evaluation and demo

First backbone candidates:

- DINOv2 ViT-B/14 for pragmatic feature quality and hardware fit
- I-JEPA ViT-H/14 checkpoint for research-aligned ablation if loading and
  runtime are tractable

Mandatory frozen baselines include DINOv2 ViT-S/14 and ViT-B/14.

## Objective

The first objective is compact pairwise JEPA-style latent prediction:

- encode context image or patches with the context encoder
- encode target image or patches with a frozen or momentum target encoder
- train a predictor/projection head to predict target latents from context
  latents
- optimize latent-space regression loss such as cosine loss or smooth L1
- track variance/covariance collapse checks
- train stable pairs as consistency targets
- keep changing pairs primarily for evaluation

Changing pairs must not be trained to collapse together. Optional hard-negative
or ranking losses require a separate ablation after the base objective works.

## Hardware boundary

Full I-JEPA reproduction is outside MVP scope. The remote GB10 tier is for
embedding export, predictor/projection training, full evals, and benchmark runs.
The MacBook Pro tier is for local demo, small slices, and artifact inspection.

## Consequences

The MVP can be serious without pretending to be a large-scale pretraining
project. If a frozen baseline wins, that is a valid result.

## Acceptance condition

This RFC is satisfied when the model code can run a tiny overfit/debug pass,
export embeddings, report collapse checks, and produce a full run directory for
the selected JEPA-style predictor.
