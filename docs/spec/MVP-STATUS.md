# DermaJEPA MVP status

Current state: Milestones 1, 2, and 3 are complete and Milestone 4 has been substantially exercised through nine primary-tier Hugging Face Jobs runs on public HAM10000 data (EXP-001 through EXP-008 plus the EXP-007/008 seed sweep). The Milestone 5 demo surface and Milestone 6 closeout artifacts remain open.

The repository does not vendor public dermatology datasets. DINOv2 and CLIP-family embedding export requires the optional `model` dependency group and a local or mounted HAM10000-style layout.

## Completed documentation

- [x] README (refreshed with current results table)
- [x] PRD
- [x] system spec
- [x] research notes
- [x] canonical MVP spec
- [x] accepted RFC stack
- [x] implementation plan
- [x] MVP schedule
- [x] Hugging Face Jobs runbook
- [x] HAM10000 JEPA run playbook

## Completed implementation milestones

### Milestone 1 — contract-first fixture pipeline

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

### Milestone 2 — public data audit and baseline path

- [x] `configs/data/ham10000.yaml` and `data/README.md`
- [x] public dataset config parsing and validation
- [x] HAM10000-style CSV normalisation into `metadata_normalized.parquet`
- [x] image availability, duplicate ID/checksum, metadata coverage, and leakage audit JSON
- [x] deterministic patient/lesion-aware split assignment
- [x] post-split stable nuisance variants
- [x] changing-pair matching by same patient, same diagnosis/site, same diagnosis, same site, then fallback
- [x] stable/changing proxy manifest and gold audit subset
- [x] pixel L2 and SSIM baseline metrics on the public-data proxy
- [x] configurable embedding model contract
- [x] DINOv2 ViT-S/14 and ViT-B/14 embedding export
- [x] OpenAI CLIP ViT-B/16 embedding export (added with EXP-006b)
- [x] open_clip-loaded embedding export covering DermLIP and BiomedCLIP via the `hf-hub:` prefix (added with EXP-007 / EXP-008)
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

These commands require local HAM10000-style raw data under `data/raw/ham10000/`; the repository does not vendor the dataset.

### Milestone 3 — JEPA-style predictor training

- [x] fixture-tier JEPA-style latent predictor over frozen image embeddings
- [x] stable-pair prediction objective; changing pairs held out for evaluation
- [x] collapse checks (prediction norm and dimension variance)
- [x] checkpoint export, training report, model card, and failure-case report
- [x] CLI `derma-jepa train --config configs/train/jepa_predictor.yaml`
- [x] linear and 2-layer MLP predictor variants under SGD and Adam (added with EXP-005 / EXP-006a)

Local acceptance command:

```bash
uv run derma-jepa train --config configs/train/jepa_predictor.yaml
```

### Milestone 4 — primary-tier evaluation and benchmark suite

Nine primary-tier runs on public HAM10000 via Hugging Face Jobs, each with a self-contained report under `docs/experiments/`:

| Run | Backbone | Predictor | Proxy variant | Test AUROC | Δ vs strongest baseline |
|---|---|---|---|---:|---:|
| [EXP-001](../experiments/EXP-001-ham10000-jepa-primary-v1.md) | DINOv2 ViT-B/14 | linear | trivial proxy | 1.000 | 0.000 |
| [EXP-002](../experiments/EXP-002-ham10000-jepa-hardened-proxy-v1.md) | DINOv2 ViT-B/14 | linear | hardened, matched eval | 0.920 | **+0.269** |
| [EXP-003](../experiments/EXP-003-ham10000-jepa-held-out-nuisance-v1.md) | DINOv2 ViT-B/14 | linear | hardened, one-family held out | 0.680 | −0.281 |
| [EXP-004](../experiments/EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md) | DINOv2 ViT-B/14 | linear | hardened, mixed + third family eval | 0.249 | −0.331 |
| [EXP-005](../experiments/EXP-005-ham10000-jepa-mlp-predictor-v1.md) | DINOv2 ViT-B/14 | MLP (underfit, SGD) | EXP-004 proxy | 0.270 | −0.310 |
| [EXP-006a](../experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md) | DINOv2 ViT-B/14 | MLP (fit, Adam) | EXP-004 proxy | 0.248 | −0.332 |
| [EXP-006b](../experiments/EXP-006b-ham10000-jepa-clip-backbone-v1.md) | OpenAI CLIP ViT-B/16 | linear | EXP-004 proxy | 0.286 | −0.294 |
| [EXP-007](../experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md) | DermLIP ViT-B/16 | linear | EXP-004 proxy | **0.944 ± 0.003** (5 seeds) | **+0.364** |
| [EXP-008](../experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) | BiomedCLIP ViT-B/16 | linear | EXP-004 proxy | 0.329 ± 0.012 (5 seeds) | −0.252 |

Plus the [EXP-007/008 seed-sweep summary](../experiments/EXP-007-008-seed-sweep-summary.md).

Milestone 4 deliverables checklist:

- [x] primary AUROC with bootstrap confidence intervals on every run
- [x] AUPRC, equal-error-rate threshold, FPR at fixed TPR (in every run's `metrics.json`)
- [x] nuisance-robustness report by family and severity (three families: `strong`, `strong_held_out`, `strong_held_out_2`)
- [x] representation-health report (prediction-norm and dimension-variance per run; collapse flag in every report)
- [x] predictor-class ablation (linear vs MLP, EXP-005 / EXP-006a)
- [x] optimiser ablation (SGD vs Adam, EXP-006a)
- [x] backbone ablation (DINOv2 / OpenAI CLIP / DermLIP / BiomedCLIP, EXP-006b / EXP-007 / EXP-008)
- [x] seed sweep on the two EXP-007 / EXP-008 headlines (5 seeds each)
- [x] qualitative failure-case JSON per run (`artifacts/reports/jepa_failure_cases.json`, `baseline_failure_cases.json`)
- [ ] runtime benchmark report (per-run wall-time tables exist; consolidated cross-run benchmark not yet written)

## Hosted-compute path

Hugging Face Jobs launchers land the training scaffold on hosted GPU compute without requiring cloud-first infrastructure. Both bundle and uv-run launchers load `scripts/hf_jobs_constraints.txt` so hosted resolves do not float.

- [x] `scripts/hf_jobs_train.sh` uv-run launcher for public GitHub refs
- [x] `scripts/hf_jobs_train_bundle.sh` private-wheel bundle launcher
- [x] `scripts/hf_jobs_constraints.txt` pinned numpy/scipy/torch/transformers/open-clip-torch versions applied via `--with` or `pip install --constraint`
- [x] `configs/data/ham10000_hf_mounted.yaml` and per-experiment variants in `configs/data/ham10000_hf_mounted_exp00*.yaml`
- [x] per-experiment launcher scripts (`scripts/hf_jobs_ham10000_exp00*.sh`) for EXP-002 through EXP-008
- [x] generic `scripts/hf_jobs_seed_sweep.sh` parameterised over a base config and `SEED`
- [x] `scripts/aggregate_seed_sweep.py` producing across-seed mean / std / 95 % CI[mean] for any list of run IDs
- [x] `derma-jepa hf-run summary` subcommand fetches a completed run by run ID from a Hub dataset/model/space repo and prints the headline metrics
- [x] fixture benchmark gate runs inside the Job before upload
- [x] `docs/runbooks/huggingface-jobs.md` covers smoke, mounted-data, pins, and the run-summary surface
- [x] `docs/runbooks/ham10000-jepa-playbook.md` covers the full primary-tier loop

## Open items toward MVP closeout

- [ ] EXP-009 — self-pretrain a DINOv2 ViT-B/14 on a non-HAM10000 dermoscopy corpus and re-run the EXP-004 recipe on top, to partition dermoscopy-domain transfer from HAM10000 image-level overlap (the central caveat on EXP-007's headline; see `docs/experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md` §6 and `docs/experiments/EXP-008-ham10000-jepa-biomedclip-backbone-v1.md` §7).
- [ ] Consolidated cross-run runtime benchmark report (Milestone 4 final deliverable).
- [ ] Local dashboard / demo surface (Milestone 5).
- [ ] MVP report and hardening (Milestone 6).

## Rule

Do not mark implementation milestones complete until the repository contains runnable or inspectable artifacts proving them.

Do not call the MVP complete until every definition-of-done item in `docs/spec/MVP-SPEC.md` is satisfied.
