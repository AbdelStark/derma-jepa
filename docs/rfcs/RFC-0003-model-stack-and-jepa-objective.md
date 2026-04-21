# RFC-0003 — model stack and jepa objective

## Status
Draft

## Problem

Choose the encoder, predictor, training objective, augmentation set, and export contract.

## Why this matters

If this decision stays fuzzy, the project will either optimize for the wrong target or bloat before the thesis is actually tested.

## Decisions to lock

- Backbone family and size
- I-JEPA-first vs temporal extension criteria
- Embedding export contract
- Training-time collapse and invariance checks

## Preferred v1 bias

Choose the smallest credible option that preserves demo speed and empirical honesty.

## Deferred items

- any move that broadens the project into a general platform
- any optimization that matters only after the first convincing demo exists
- any expansion in data/model size that does not materially change the first evaluation story

## Acceptance condition

This RFC is complete only when a builder could implement the next phase without guessing what the project is actually trying to prove.
