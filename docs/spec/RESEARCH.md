# DermaJEPA — research notes

## Project domain

dermatology imaging and longitudinal change detection

## Central research question

A JEPA-style encoder can learn a lesion representation where stable lesions remain close under nuisance variation and meaningful lesion evolution appears as a structured latent trajectory departure.

## Data sources under consideration

- ISIC Archive
- HAM10000
- PAD-UFES-20
- Optional same-lesion subset curation if enough repeated examples exist

## Questions to answer before implementation grows

1. What is the smallest task framing that still tests the thesis honestly?
2. What is the strongest cheap baseline?
3. What nuisance or confounders matter most in practice?
4. What would count as a fake win here?
5. What result would actually persuade a skeptical technical reader?

## Research anti-patterns

- overclaiming beyond the chosen task
- broadening the problem until the evaluation becomes vague
- substituting visual polish for empirical clarity
- comparing only against weak baselines
- letting the demo narrative outrun the measured result

## Research spikes

- `docs/research-spikes/2026-04-21-ml-intern-and-medical-sam.md` — evaluation of Hugging Face `ml-intern` and whether a SAM-style medical segmentation workflow should be used as a DermaJEPA sidecar
