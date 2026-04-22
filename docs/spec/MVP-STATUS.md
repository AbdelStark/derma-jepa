# DermaJEPA MVP status

Current state: Milestones 1 and 2 are complete, the Milestone 3 JEPA-style
predictor scaffold runs locally and on Hugging Face Jobs for the fixture tier,
and the hosted-compute path has pinned dependency versions, a mounted-data
HAM10000 config, and a command for fetching completed runs from the Hub.

The repository still does not vendor public dermatology datasets. DINOv2
embedding export requires the optional `model` dependency group and a local or
mounted HAM10000-style layout.

## Completed documentation

- [x] README
- [x] PRD
- [x] system spec
- [x] research notes
- [x] canonical MVP spec
- [x] accepted RFC stack
- [x] implementation plan
- [x] MVP schedule
- [x] Hugging Face Jobs runbook

## Completed implementation milestones

### Milestone 1 - contract-first fixture pipeline

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

### Milestone 2 - public data audit and baseline path

- [x] `configs/data/ham10000.yaml` and `data/README.md`
- [x] public dataset config parsing and validation
- [x] HAM10000-style CSV normalization into `metadata_normalized.parquet`
- [x] image availability, duplicate ID/checksum, metadata coverage, and leakage
      audit JSON
- [x] deterministic patient/lesion-aware split assignment
- [x] post-split stable nuisance variants
- [x] changing-pair matching by same patient, same diagnosis/site, same
      diagnosis, same site, then fallback
- [x] stable/changing proxy manifest and gold audit subset
- [x] pixel L2 and SSIM baseline metrics on the public-data proxy
- [x] configurable embedding model contract
- [x] DINOv2 ViT-S/14 and ViT-B/14 embedding export
- [x] deterministic embedding backend for local contract tests
- [x] embedding-cosine baseline integration into `baseline_metrics.json`
- [x] baseline failure-case templates
- [x] fixture benchmark gate requires embedding index and failure-case artifacts

Public-data local acceptance commands:

```bash
uv run derma-jepa data audit --config configs/data/ham10000.yaml
uv run derma-jepa manifest build --config configs/data/ham10000.yaml
uv run derma-jepa embed --config configs/data/ham10000.yaml
uv run derma-jepa baseline eval --config configs/data/ham10000.yaml
```

These commands require local HAM10000-style raw data under `data/raw/ham10000/`;
the repository does not vendor the dataset.

### Milestone 3 scaffold - JEPA-style predictor training

- [x] fixture-tier JEPA-style latent predictor over frozen image embeddings
- [x] stable-pair prediction objective; changing pairs held out for evaluation
- [x] collapse checks (prediction norm and dimension variance)
- [x] checkpoint export, training report, model card, and failure-case report
- [x] CLI `derma-jepa train --config configs/train/jepa_predictor.yaml`

Local acceptance command:

```bash
uv run derma-jepa train --config configs/train/jepa_predictor.yaml
```

## Hosted-compute path

Hugging Face Jobs launchers land the training scaffold on hosted GPU/CPU
compute without requiring cloud-first infrastructure. Both launchers load
`scripts/hf_jobs_constraints.txt` so hosted resolves do not float.

- [x] `scripts/hf_jobs_train.sh` uv-run launcher for public GitHub refs
- [x] `scripts/hf_jobs_train_bundle.sh` private-wheel bundle launcher
- [x] `scripts/hf_jobs_constraints.txt` pinned numpy/scipy/torch/transformers
      versions applied via `--with` or `pip install --constraint`
- [x] `configs/data/ham10000_hf_mounted.yaml` mounted-data HAM10000 config
      whose paths match `HF_JOBS_VOLUME="hf://datasets/<ns>/<repo>:/data:ro"`
- [x] `derma-jepa hf-run summary` subcommand fetches a completed run by run ID
      from a Hub dataset/model/space repo and prints the headline metrics
- [x] fixture benchmark gate runs inside the Job before upload
- [x] `docs/runbooks/huggingface-jobs.md` covers smoke, mounted-data, pins,
      and the run-summary surface

## Remaining / in progress

- [x] public data audit
- [x] public dataset indexing/download workflow
- [x] leakage-controlled primary manifest
- [x] DINOv2 baseline implementation
- [x] primary-tier embedding export contract
- [x] initial primary-tier evaluation reports
- [x] initial primary-tier model card
- [x] JEPA-style predictor training scaffold
- [x] JEPA-style downstream drift scoring path
- [x] representation health checks
- [x] hosted-compute path for fixture-tier JEPA training
- [ ] first real public-data JEPA training run on HAM10000 via HF Jobs
- [ ] dermatology-supervised baseline investigation
- [ ] nuisance robustness benchmark
- [ ] local dashboard/demo surface

## Rule

Do not mark implementation milestones complete until the repository contains
runnable or inspectable artifacts proving them.

Do not call the MVP complete until every definition-of-done item in
`docs/spec/MVP-SPEC.md` is satisfied.
