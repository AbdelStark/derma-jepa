# Hugging Face Jobs training

Use this path when the local fixture train command is working and the next run
needs hosted GPU compute.

## Prerequisites

- Hugging Face account with Jobs access and billing/prepaid credits configured.
- Local `hf` CLI installed and authenticated:

```bash
brew install hf
hf auth login
```

## Fixture smoke on Hugging Face

This launches the same fixture training config used by the local smoke:

```bash
./scripts/hf_jobs_train.sh
```

Defaults:

- Git ref: `main`
- config: `configs/train/jepa_predictor.yaml`
- hardware: `a10g-small`
- timeout: `2h`
- run output root inside the Job: `outputs/runs`

To print the exact `hf jobs uv run` command without launching compute:

```bash
HF_JOBS_DRY_RUN=1 ./scripts/hf_jobs_train.sh
```

## Upload completed run artifacts

Set a Hub dataset repo for run outputs:

```bash
HF_OUTPUT_REPO_ID="$HF_USER/derma-jepa-runs" \
HF_OUTPUT_REPO_TYPE=dataset \
./scripts/hf_jobs_train.sh
```

The launcher passes `HF_TOKEN` as a Job secret when `HF_OUTPUT_REPO_ID` is set.

## Public-data or primary-tier run

Use a config URL whose dataset paths match the mounted volume:

```bash
DERMA_JEPA_CONFIG_URL="https://raw.githubusercontent.com/AbdelStark/derma-jepa/main/configs/data/ham10000.yaml" \
HF_JOBS_VOLUME="hf://datasets/<namespace>/<dataset-repo>:/data:ro" \
HF_JOBS_FLAVOR=a100-large \
HF_JOBS_TIMEOUT=12h \
./scripts/hf_jobs_train.sh
```

For public dermatology datasets, prefer a separate config whose `metadata_csv`
and `image_roots` point at the mounted paths. Do not upload raw patient or
personal images. The MVP remains a research artifact, not a diagnostic system.

## Useful overrides

- `DERMA_JEPA_REF`: Git ref for both package install and raw script/config URLs.
- `DERMA_JEPA_CONFIG_URL`: raw YAML config URL used by the Job entrypoint.
- `DERMA_JEPA_RUN_ID`: run ID written into artifacts.
- `HF_JOBS_FLAVOR`: hardware flavor, for example `a10g-small` or `a100-large`.
- `HF_JOBS_TIMEOUT`: max Job runtime, for example `2h` or `12h`.
- `HF_JOBS_NAMESPACE`: user or organization namespace to bill/run under.
- `HF_JOBS_VOLUME`: one volume mount, for example `hf://datasets/org/ds:/data:ro`.
- `HF_JOBS_DETACH=1`: start the Job and return immediately.
