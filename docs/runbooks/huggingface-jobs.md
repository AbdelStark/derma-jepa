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

## Fixture smoke on Hugging Face for a private GitHub repo

This builds a local wheel, embeds the wheel plus the local config into the Job
script, and avoids cloning GitHub from inside Hugging Face:

```bash
./scripts/hf_jobs_train_bundle.sh
```

The fixture Job runs training, then runs the fixture benchmark gate before
uploading artifacts.

To print the exact `hf jobs uv run` command without launching compute:

```bash
HF_JOBS_DRY_RUN=1 ./scripts/hf_jobs_train_bundle.sh
```

## Fixture smoke on Hugging Face for a public GitHub repo

This launches from raw GitHub URLs and installs the package from the Git ref:

```bash
./scripts/hf_jobs_train.sh
```

Defaults:

- Git ref: `main`
- config: `configs/train/jepa_predictor.yaml`
- hardware: `cpu-upgrade`
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
./scripts/hf_jobs_train_bundle.sh
```

The launcher passes `HF_TOKEN` as a Job secret when `HF_OUTPUT_REPO_ID` is set.

## Public-data or primary-tier run

Use the mounted-data HAM10000 config and a volume that resolves the same paths
inside the Job. `configs/data/ham10000_hf_mounted.yaml` is the checked-in
counterpart to `configs/data/ham10000.yaml` with `/data` paths that match the
volume mount:

```bash
DERMA_JEPA_CONFIG_PATH=configs/data/ham10000_hf_mounted.yaml \
HF_JOBS_VOLUME="hf://datasets/<namespace>/<dataset-repo>:/data:ro" \
HF_JOBS_FLAVOR=a100-large \
HF_JOBS_TIMEOUT=12h \
./scripts/hf_jobs_train_bundle.sh
```

The mounted dataset repo must contain `HAM10000_metadata.csv` at the root and
images under `HAM10000_images_part_1/` and `HAM10000_images_part_2/`. Do not
upload raw patient or personal images. The MVP remains a research artifact,
not a diagnostic system.

## Pinned hosted dependencies

Both launchers apply `scripts/hf_jobs_constraints.txt` so hosted runs do not
float to whatever numpy/scipy/torch/transformers happens to be newest when the
Job starts:

- `hf_jobs_train.sh` adds each pin as a `--with pkg==ver` flag so `uv run`
  resolves the ephemeral environment against them.
- `hf_jobs_train_bundle.sh` embeds the file into the Job script and passes it
  to `pip install --constraint` during bootstrap.

Keep the pins aligned with `uv.lock` when you upgrade. Override the file with
`DERMA_JEPA_PINS_FILE=/path/to/custom.txt` when experimenting.

## Fetch and summarize a completed run

Run outputs are uploaded to the Hub under `path_in_repo = run_id`. To pull the
artifacts back and see the headline numbers:

```bash
uv run derma-jepa hf-run summary \
  --repo-id "$HF_USER/derma-jepa-runs" \
  --run-id hf-jepa-fixture-20260422-120000
```

Use `--json` for machine-readable output, `--dest` to pick a local cache root,
`--path-in-repo` if the upload used a non-default subfolder, and `--revision`
for a specific Hub commit. The command requires `huggingface-hub` to be
installed locally.

## Useful overrides

- `DERMA_JEPA_REF`: Git ref for both package install and raw script/config URLs.
- `DERMA_JEPA_CONFIG_PATH`: local config path for the private bundle launcher.
- `DERMA_JEPA_CONFIG_URL`: raw YAML config URL used by the Job entrypoint.
- `DERMA_JEPA_RUN_ID`: run ID written into artifacts.
- `DERMA_JEPA_INSTALL_EXTRAS`: optional package extras installed from the wheel.
- `HF_JOBS_FLAVOR`: hardware flavor, for example `cpu-upgrade` or `a100-large`.
- `HF_JOBS_TIMEOUT`: max Job runtime, for example `2h` or `12h`.
- `HF_JOBS_NAMESPACE`: user or organization namespace to bill/run under.
- `HF_JOBS_VOLUME`: one volume mount, for example `hf://datasets/org/ds:/data:ro`.
- `HF_JOBS_DETACH=1`: start the Job and return immediately.
