#!/usr/bin/env bash
#
# Launch DermaJEPA EXP-008 — BiomedCLIP (ViT-B/16 CLIP-trained on
# PMC-15M PubMed Central figure-caption pairs, MIT licence) backbone
# swap on the EXP-004 proxy. Partitions EXP-007's win between "any
# medical pretraining unlocks the proxy" and "specifically HAM10000-
# contaminated pretraining unlocks the proxy."
#
# See docs/experiments/EXP-007-ham10000-jepa-dermlip-backbone-v1.md §7
# for the decision table and contamination-disambiguation rationale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

: "${DERMA_JEPA_HAM10000_REPO:=abdelstark/ham10000}"
: "${DERMA_JEPA_RUN_ID:=ham10000-hf-biomedclip-primary-exp008-$(date -u +%Y%m%d-%H%M%S)}"
: "${HF_JOBS_FLAVOR:=a10g-large}"
: "${HF_JOBS_TIMEOUT:=12h}"
: "${HF_JOBS_DETACH:=1}"

export DERMA_JEPA_CONFIG_PATH="configs/data/ham10000_hf_mounted_exp008.yaml"
export DERMA_JEPA_INSTALL_EXTRAS="model"
export HF_JOBS_VOLUME="hf://datasets/${DERMA_JEPA_HAM10000_REPO}:/data:ro"
export HF_JOBS_FLAVOR
export HF_JOBS_TIMEOUT
export HF_JOBS_DETACH
export DERMA_JEPA_RUN_ID

printf 'Launching DermaJEPA EXP-008 (BiomedCLIP / EXP-004 proxy)\n'
printf '  run_id        = %s\n' "$DERMA_JEPA_RUN_ID"
printf '  dataset mount = %s\n' "$HF_JOBS_VOLUME"
printf '  flavor        = %s\n' "$HF_JOBS_FLAVOR"
printf '  timeout       = %s\n' "$HF_JOBS_TIMEOUT"
printf '  config        = %s\n' "$DERMA_JEPA_CONFIG_PATH"
printf '\n'

exec "$SCRIPT_DIR/hf_jobs_train_bundle.sh" "$@"
