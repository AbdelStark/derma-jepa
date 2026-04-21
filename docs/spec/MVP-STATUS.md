# DermaJEPA MVP status

Current state: MVP specification locked; implementation not started.

## Completed

- [x] README
- [x] PRD
- [x] system spec
- [x] research notes
- [x] canonical MVP spec
- [x] accepted RFC stack
- [x] implementation plan
- [x] MVP schedule

## Active next milestone

- [ ] Milestone 1: contract-first fixture pipeline

Milestone 1 must produce:

- [ ] package scaffold and `pyproject.toml`
- [ ] CLI skeleton
- [ ] manifest schema and validation
- [ ] synthetic/tiny fixture dataset
- [ ] preprocessing profile implementation
- [ ] pixel/SSIM baseline on fixtures
- [ ] AUROC and bootstrap CI metric wrapper
- [ ] run directory writer
- [ ] demo export JSON for one fixture case
- [ ] CI fixture pipeline

## Not started

- [ ] data audit
- [ ] public dataset indexing/download workflow
- [ ] leakage-controlled primary manifest
- [ ] DINOv2 baseline implementation
- [ ] dermatology-supervised baseline investigation
- [ ] JEPA-style predictor training scaffold
- [ ] embedding export contract
- [ ] downstream drift scoring path
- [ ] nuisance robustness benchmark
- [ ] representation health checks
- [ ] local dashboard/demo surface
- [ ] evaluation reports
- [ ] model card
- [ ] README reproduction update

## Rule

Do not mark implementation milestones complete until the repository contains
runnable or inspectable artifacts proving them.

Do not call the MVP complete until every definition-of-done item in
`docs/spec/MVP-SPEC.md` is satisfied.
