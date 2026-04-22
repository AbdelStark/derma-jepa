# HAM10000 JEPA run playbook

Operational guide for running, interpreting, and archiving DermaJEPA training
Jobs on public HAM10000 data. This playbook is the source of truth for the
research engineering loop; its outputs feed future writeups, educational
content, and results packaging.

## 0. What we are actually measuring

DermaJEPA's primary MVP claim is bounded and spec-locked
(`docs/spec/MVP-SPEC.md`):

> On a leakage-controlled longitudinal-proxy task, does a JEPA-style latent
> predictor beat the strongest cheap baseline (pixel L2, SSIM, DINOv2 cosine)
> at separating "stable" from "changing" image pairs derived from HAM10000?

Three allowed outcomes:

1. JEPA predictor beats the strongest baseline by some AUROC delta with CI.
2. JEPA predictor matches the strongest baseline (delta ~ 0, including both
   at ceiling — "DINOv2 already solves the task").
3. JEPA predictor loses.

All three are legitimate research results. We do not tune thresholds,
cherry-pick cases, or weaken baselines to rescue outcome (3).

## 1. Prerequisites

| Resource | Where | Check |
|---|---|---|
| Hugging Face account with Jobs access | `hf auth whoami` | returns a user |
| `HF_TOKEN` with read on HAM10000 repo + write on runs repo | `.env` | `echo $HF_TOKEN` non-empty |
| Private HAM10000 dataset on the Hub | `hf://datasets/<ns>/ham10000` | `HAM10000_metadata.csv`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/` present |
| Runs output dataset on the Hub | `HF_OUTPUT_REPO_ID` | repo exists; token has write |
| Local repo clean, tests green | `git status`, `uv run pytest -q` | clean + all pass |

### Uploading HAM10000 to the Hub (one-time)

Obtain raw HAM10000 from Harvard Dataverse (DOI `10.7910/DVN/DBW86T`). Review
access terms before use. The image extraction typically yields:

```
HAM10000_images_part_1/ISIC_*.jpg       (5000 files)
HAM10000_images_part_2/ISIC_*.jpg       (5015 files)
HAM10000_metadata                       (CSV, no extension)
```

Create a **private** dataset and upload the three paths, renaming the
metadata file to add `.csv`:

```bash
hf repos create <ns>/ham10000 --repo-type dataset --private
hf upload <ns>/ham10000 data/raw/ham10000/HAM10000_metadata HAM10000_metadata.csv --repo-type dataset
hf upload <ns>/ham10000 data/raw/ham10000/HAM10000_images_part_1 HAM10000_images_part_1 --repo-type dataset
hf upload <ns>/ham10000 data/raw/ham10000/HAM10000_images_part_2 HAM10000_images_part_2 --repo-type dataset
```

Verify:

```bash
uv run --with "huggingface-hub>=1.0" python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('<ns>/ham10000', repo_type='dataset')
print('total files:', len(files))
print('metadata:', 'HAM10000_metadata.csv' in files)
print('part1:', sum(1 for f in files if f.startswith('HAM10000_images_part_1/')))
print('part2:', sum(1 for f in files if f.startswith('HAM10000_images_part_2/')))
"
```

Expect: 10017 files, metadata present, 5000 + 5015 images.

## 2. Smoke first, always

Before any primary-tier run:

```bash
DERMA_JEPA_CONFIG_PATH=configs/data/ham10000_hf_mounted_smoke.yaml \
DERMA_JEPA_INSTALL_EXTRAS=model \
HF_JOBS_VOLUME="hf://datasets/<ns>/ham10000:/data:ro" \
HF_JOBS_FLAVOR=a10g-small \
HF_JOBS_TIMEOUT=1h \
HF_JOBS_DETACH=1 \
DERMA_JEPA_RUN_ID=ham10000-hf-smoke-$(date -u +%Y%m%d-%H%M%S) \
./scripts/hf_jobs_train_bundle.sh
```

Smoke purpose:

- Confirms the dataset mount resolves.
- Confirms `[model]` extras install (torch/timm/transformers/cudnn land).
- Confirms `config.fixture is None` and the public-data manifest builder
  runs against `/data`.
- Confirms the uploader writes to `hf://datasets/$HF_OUTPUT_REPO_ID/<run_id>/`.
- Smoke `tier` must be `public`, not `fixture`. A `fixture` tier means the
  config override did not land — stop and fix before spending primary-tier
  compute.

A smoke with `auroc: 1.0000` and `delta_vs_baseline: +0.0000` at 100 pairs
is expected — the proxy task is trivially separable at tiny scale. It does
**not** mean the primary run will also ceiling.

## 3. Primary-tier launch

Single-command launcher lives at `scripts/hf_jobs_ham10000_primary.sh`. It
pins the config path, extras, and mount to the canonical values and
defaults to `a100-large` + `12h`:

```bash
./scripts/hf_jobs_ham10000_primary.sh
```

Override:

```bash
HF_JOBS_FLAVOR=a10g-large HF_JOBS_TIMEOUT=6h \
DERMA_JEPA_RUN_ID=ham10000-hf-dinov2-primary-v1 \
./scripts/hf_jobs_ham10000_primary.sh
```

Dry-run before spending:

```bash
HF_JOBS_DRY_RUN=1 ./scripts/hf_jobs_ham10000_primary.sh
```

The dry-run output must contain:

- `--volume hf://datasets/<ns>/ham10000:/data:ro`
- `--install-extras model`
- the expected `--flavor`, `--timeout`, and `--run-id`
- upload env: `HF_OUTPUT_REPO_ID`, `HF_OUTPUT_REPO_TYPE=dataset`,
  `HF_OUTPUT_PATH=<run-id>`

## 4. Watching the Job

```bash
hf jobs inspect <job-id>             # status snapshot
hf jobs logs <job-id> --follow       # live logs
```

Job phases to expect:

1. `SCHEDULING` — 30–120s while the GPU provisions.
2. `RUNNING` — wheel + pinned deps install (~60–90s for model extras,
   mostly torch + cudnn).
3. Training — public manifest build, embedding export (DINOv2 ViT-S/14 and
   ViT-B/14), cheap baselines, JEPA predictor fit.
4. Upload — run directory pushes to `hf://datasets/$HF_OUTPUT_REPO_ID/<run-id>/`.

## 5. Pulling and summarizing results

```bash
uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary \
  --repo-id "$HF_OUTPUT_REPO_ID" \
  --run-id <run-id>
```

Key fields to read:

| Field | What it tells you |
|---|---|
| `tier` | `public` for real HAM10000, `fixture` for synthetic |
| `model_id` | config-locked predictor identifier |
| `primary_score` (AUROC) | JEPA predictor test AUROC |
| `strongest_baseline` | name + AUROC of the best pixel/SSIM/embedding baseline |
| `delta_vs_baseline` | JEPA AUROC minus strongest baseline AUROC |
| `collapsed` | must be `False`; `True` invalidates the run |
| `interpretation` | config-generated positive/negative/ceiling wording |
| `runtime_seconds` | linear predictor fit time, not end-to-end Job time |

Run output lands in `outputs/hf-runs/<run-id>/` with the full artifact
contract from `docs/spec/MVP-SPEC.md` §14 (config, manifests, metrics,
baseline_metrics, model_card, embeddings, plots, demo cases, logs).

## 6. Interpreting the delta

| Delta AUROC | Collapse? | Interpretation |
|---|---|---|
| >= +0.05 | False | Primary-tier positive. Report with bootstrap CI. |
| 0 < delta < 0.05 | False | Marginal positive. Report delta + CI honestly. |
| ~ 0 with both at ceiling | False | "Frozen DINOv2 already solves the task" — legitimate negative per MVP spec §21 risk check. |
| < 0 | False | Honest negative: JEPA objective did not help on this proxy. |
| Any | True | Run is invalid — JEPA representation collapsed. Debug before reinterpreting. |

**Do not** tune thresholds, retry with different seeds until positive,
remove baselines, or cherry-pick splits. The MVP failure policy
(`docs/spec/MVP-SPEC.md` §20) explicitly forbids these.

## 7. Archiving for writeups

Everything future educational content / papers need is already in the run
directory. Canonical mapping:

| Writeup artifact | Source |
|---|---|
| Headline metrics table | `metrics.json` (`result.jepa_style_model.metrics`, `result.strongest_baseline`) |
| Baseline comparison | `baseline_metrics.json` |
| Score distributions | `artifacts/plots/jepa_score_histogram.png` and `baseline_score_histogram.png` |
| Failure cases | `artifacts/reports/jepa_failure_cases.json`, `baseline_failure_cases.json` |
| Training dynamics | `artifacts/reports/jepa_training_report.json` (epochs, train/val loss) |
| Model card | `model_card.md` |
| Representation health | `metrics.representation_health` (norms, variances, collapse flag) |
| Reproducibility | `config.yaml`, `environment.txt`, the run's `run_id` |

When citing a run in documentation, always include: `run_id`, `tier`,
`model_id`, input embedding model, DINOv2 checkpoint revision (from
`config.yaml`), bootstrap CI, and the Hub path of the run directory.

## 8. Cost / time envelope

From observed smoke timing:

- Bundle build + upload: ~30s local.
- Scheduling: 30–120s.
- Dependency install (with `[model]` extras): 60–90s.
- Embedding export (10015 images, ViT-B/14): minutes on A100, ~10 minutes
  on A10G.
- Manifest + baselines + JEPA fit: seconds to low minutes.
- Upload of run dir: ~30s.

Rule of thumb: primary-tier run completes in well under 1h on `a100-large`.
The 12h timeout is a safety net, not an expected runtime. Bill only what
the Job actually uses.

## 9. Safety and licensing reminders

- The repo never vendors HAM10000 images; only the Hub dataset does, and
  that repo is private.
- `HF_TOKEN` goes to the Job as a secret and is used for both dataset read
  and artifact write. Scope it to the minimum namespaces that need it.
- Run outputs must not contain raw patient or personal images. Only the
  deterministic nuisance-variant fixture images land under
  `fixture/images/` and only in the fixture tier.
- All run-level copy (model card, reports) is research-only:
  **not diagnostic, not medical advice, not validated for patient use.**

## 10. When something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `tier: fixture` on a HAM10000 run | `DERMA_JEPA_CONFIG_PATH` override did not land (usually shell paste eating backslash-continuations) | Use the named launcher `hf_jobs_ham10000_primary.sh` or `export` vars one per line |
| Pip install log lacks `torch` / `timm` / `transformers` | `DERMA_JEPA_INSTALL_EXTRAS` unset | Set it via `.env` or as the first inline var |
| `split <X> has N available images but M pairs per label were requested` | `max_records` too small for requested pairs given split fractions | Raise `max_records` so each split has at least `stable_pairs_per_split + changing_pairs_per_split` images |
| Exit code 137 | OOM or Job-level kill | Check logs for a Python traceback; if none, drop batch_size or pick a larger flavor |
| `RuntimeError: huggingface-hub is required` from `hf-run summary` | local venv does not have huggingface-hub | Use `uv run --with "huggingface-hub>=1.0" derma-jepa hf-run summary …` |
| Job hangs in SCHEDULING for >5m | Hub capacity or account limits | Re-inspect; if still stuck, cancel and retry or switch flavor |

This file is edited in place as the playbook evolves. Real changes ship
through PRs and the `docs/runbooks/huggingface-jobs.md` file remains the
infra-tooling reference.
