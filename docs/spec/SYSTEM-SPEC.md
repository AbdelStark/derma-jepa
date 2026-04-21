# DermaJEPA — system spec

## 1. Architecture summary

The system should be built in six layers:
1. raw data ingestion
2. sample/window construction
3. JEPA encoder + predictor training
4. downstream scoring / decode / forecast path
5. demo-facing application surface
6. evaluation and artifact export

## 2. System constraints

- single-machine friendly for development
- bounded training budget
- explicit baseline before large model work
- deterministic enough to reproduce demo artifacts
- narrow enough to evaluate honestly

## 3. Data contract

The project must define:
- what one sample/window/trajectory means
- what the context input is
- what the target representation is
- what nuisance variables must be preserved vs suppressed
- how train/validation/test boundaries prevent leakage

## 4. Model contract

The project must specify:
- encoder backbone and parameter scale
- predictor head architecture
- latent representation dimensions
- training objective and regularization
- embedding export and downstream interface

## 5. Evaluation contract

The system must expose:
- one primary task metric
- at least one nuisance-robustness or invariance check
- at least one cheap, credible baseline
- at least one human-inspectable artifact

## 6. Demo contract

The demo must:
- show the thesis in one interaction or one short walkthrough
- produce screenshot/video-friendly outputs
- avoid hidden manual data cleanup in the live path
- make failure cases visible rather than silently hiding them

## 7. Implementation principle

Build the narrowest meaningful path first. The first version must prove the thesis, not become a platform.
