# DermaJEPA MVP status

Current state: Milestone 1 complete; fixture-tier implementation is runnable.

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

- [ ] Milestone 2: public data audit and baseline path

Milestone 2 must produce:

- [ ] `data/README.md` with source, license/access, expected files, checksums
      where possible, and citations
- [ ] HAM10000/ISIC indexing or download instructions
- [ ] normalized metadata tables
- [ ] leakage audit with patient/lesion/source/duplicate checks
- [ ] stable/changing proxy manifest
- [ ] gold audit subset
- [ ] pixel/SSIM baseline report on the public-data proxy
- [ ] DINOv2 ViT-S/14 and ViT-B/14 embedding export
- [ ] DINOv2 baseline report
- [ ] initial failure-case templates

## Not started

- [ ] public data audit
- [ ] public dataset indexing/download workflow
- [ ] leakage-controlled primary manifest
- [ ] DINOv2 baseline implementation
- [ ] dermatology-supervised baseline investigation
- [ ] JEPA-style predictor training scaffold
- [ ] primary-tier embedding export contract
- [ ] JEPA-style downstream drift scoring path
- [ ] nuisance robustness benchmark
- [ ] representation health checks
- [ ] local dashboard/demo surface
- [ ] primary-tier evaluation reports
- [ ] primary-tier model card

## Rule

Do not mark implementation milestones complete until the repository contains
runnable or inspectable artifacts proving them.

Do not call the MVP complete until every definition-of-done item in
`docs/spec/MVP-SPEC.md` is satisfied.
