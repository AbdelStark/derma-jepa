# DermaJEPA MVP status

Current state: Milestone 1 complete; Milestone 2 public-data audit, manifest,
embedding export, and baseline-report path is runnable against local
HAM10000-style files. DINOv2 export requires the optional model dependency group
and local raw data.

## Completed

- [x] README
- [x] PRD
- [x] system spec
- [x] research notes
- [x] canonical MVP spec
- [x] accepted RFC stack
- [x] implementation plan
- [x] MVP schedule
- [x] Milestone 1: contract-first fixture pipeline
- [x] Milestone 2: public data audit and baseline path

## Completed implementation milestone

- [x] Milestone 1: contract-first fixture pipeline

Milestone 1 produced:

- [x] package scaffold and `pyproject.toml`
- [x] Typer CLI with the locked MVP command surface
- [x] manifest schema and validation
- [x] deterministic synthetic fixture dataset generator
- [x] preprocessing profile implementation
- [x] pixel L2 and SSIM baselines on fixtures
- [x] AUROC, AUPRC, threshold, and bootstrap CI metric wrapper
- [x] run directory writer with fixture-tier artifact validation
- [x] demo export JSON plus static HTML for one fixture run
- [x] CI fixture pipeline

Local acceptance command:

```bash
uv run derma-jepa fixture pipeline --config configs/manifest/fixture.yaml
```

## Active next milestone

- [ ] Milestone 3: JEPA-style predictor training

Milestone 2 produced:

- [x] `data/README.md` with source, license/access, expected files, checksums
      where possible, and citations
- [x] HAM10000 indexing or download instructions
- [x] normalized metadata tables
- [x] leakage audit with patient/lesion/source/duplicate checks
- [x] stable/changing proxy manifest
- [x] gold audit subset
- [x] pixel/SSIM baseline report on the public-data proxy
- [x] DINOv2 ViT-S/14 and ViT-B/14 embedding export
- [x] DINOv2 baseline report
- [x] initial failure-case templates

## Remaining / in progress

- [x] public data audit
- [x] public dataset indexing/download workflow
- [x] leakage-controlled primary manifest
- [x] DINOv2 baseline implementation
- [x] primary-tier embedding export contract
- [x] initial primary-tier evaluation reports
- [x] initial primary-tier model card
- [ ] dermatology-supervised baseline investigation
- [ ] JEPA-style predictor training scaffold
- [ ] JEPA-style downstream drift scoring path
- [ ] nuisance robustness benchmark
- [ ] representation health checks
- [ ] local dashboard/demo surface

## Completed implementation slice

- [x] Milestone 2 slice: HAM10000-style audit, proxy manifest, and cheap
      baseline path

This slice produced:

- [x] `configs/data/ham10000.yaml`
- [x] `data/README.md`
- [x] public dataset config parsing and validation
- [x] HAM10000-style CSV normalization into `metadata_normalized.parquet`
- [x] image availability, duplicate ID/checksum, metadata coverage, and leakage
      audit JSON
- [x] deterministic patient/lesion-aware split assignment
- [x] post-split stable nuisance variants
- [x] changing-pair matching by same patient, same diagnosis/site, same
      diagnosis, same site, then fallback
- [x] public-tier pixel L2 and SSIM baseline metrics
- [x] tests covering audit, missing-image failure, manifest validation, leakage
      constraints, and baseline artifact generation

## Completed implementation milestone

- [x] Milestone 2: public data audit and model-baseline path

This milestone added:

- [x] configurable embedding model contract
- [x] optional DINOv2 ViT-S/14 and ViT-B/14 export path
- [x] deterministic embedding backend for local contract tests
- [x] embedding index artifacts under `artifacts/embeddings/`
- [x] embedding-cosine baseline integration into `baseline_metrics.json`
- [x] manual gold-audit subset under `artifacts/reports/gold_audit_subset.csv`
- [x] baseline failure-case templates under
      `artifacts/reports/baseline_failure_cases.json`
- [x] fixture benchmark gate updated to require embedding index and failure-case
      artifacts

Public-data local acceptance commands:

```bash
uv run derma-jepa data audit --config configs/data/ham10000.yaml
uv run derma-jepa manifest build --config configs/data/ham10000.yaml
uv run derma-jepa baseline eval --config configs/data/ham10000.yaml
```

These commands require local HAM10000-style raw data under `data/raw/ham10000/`;
the repository does not vendor the dataset.

## Rule

Do not mark implementation milestones complete until the repository contains
runnable or inspectable artifacts proving them.

Do not call the MVP complete until every definition-of-done item in
`docs/spec/MVP-SPEC.md` is satisfied.
