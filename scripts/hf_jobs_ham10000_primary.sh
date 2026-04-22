#!/usr/bin/env bash
#
# Launch the DermaJEPA primary-tier HAM10000 run on Hugging Face Jobs.
#
# Uses configs/data/ham10000_hf_mounted.yaml (ViT-B/14 + ViT-S/14 DINOv2
# embeddings, 1000 pairs per split, full 10015-image dataset, 200 epochs).
# Defaults to `a10g-large` on a 12h ceiling because A100 availability on
# the Hub Jobs queue is inconsistent; override to `a100-large` when it is.
#
# Required in .env or the calling shell:
#   HF_TOKEN                 with read on the HAM10000 dataset repo and write
#                            on the runs dataset repo
#   HF_OUTPUT_REPO_ID        target dataset repo for uploaded run artifacts
#
# Useful overrides (env vars):
#   DERMA_JEPA_RUN_ID        run identifier, also the Hub subfolder name
#   DERMA_JEPA_HAM10000_REPO namespace/name of the HAM10000 dataset on the Hub
#   HF_JOBS_FLAVOR           a10g-large (default), a100-large, etc.
#   HF_JOBS_TIMEOUT          e.g. 6h, 12h (default)
#   HF_JOBS_DETACH           0 to stream logs, 1 (default) to return Job ID
#
# This wraps scripts/hf_jobs_train_bundle.sh so the private-wheel bundle
# launcher still owns dependency pins and config embedding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

: "${DERMA_JEPA_HAM10000_REPO:=abdelstark/ham10000}"
: "${DERMA_JEPA_RUN_ID:=ham10000-hf-dinov2-primary-$(date -u +%Y%m%d-%H%M%S)}"
: "${HF_JOBS_FLAVOR:=a10g-large}"
: "${HF_JOBS_TIMEOUT:=12h}"
: "${HF_JOBS_DETACH:=1}"

export DERMA_JEPA_CONFIG_PATH="configs/data/ham10000_hf_mounted.yaml"
export DERMA_JEPA_INSTALL_EXTRAS="model"
export HF_JOBS_VOLUME="hf://datasets/${DERMA_JEPA_HAM10000_REPO}:/data:ro"
export HF_JOBS_FLAVOR
export HF_JOBS_TIMEOUT
export HF_JOBS_DETACH
export DERMA_JEPA_RUN_ID

printf 'Launching DermaJEPA primary-tier HAM10000 run\n'
printf '  run_id        = %s\n' "$DERMA_JEPA_RUN_ID"
printf '  dataset mount = %s\n' "$HF_JOBS_VOLUME"
printf '  flavor        = %s\n' "$HF_JOBS_FLAVOR"
printf '  timeout       = %s\n' "$HF_JOBS_TIMEOUT"
printf '  config        = %s\n' "$DERMA_JEPA_CONFIG_PATH"
printf '\n'

exec "$SCRIPT_DIR/hf_jobs_train_bundle.sh" "$@"
