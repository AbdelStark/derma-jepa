#!/usr/bin/env bash
#
# Launch DermaJEPA EXP-007 — DermLIP PanDerm (dermoscopy-pretrained
# ViT-B/16 via open_clip) backbone swap on the EXP-004 proxy. Tests
# whether a backbone pretrained on ~3M dermatology images lifts the
# below-random inversion observed across EXP-004/005/006a (DINOv2) and
# EXP-006b (OpenAI CLIP) on strong_held_out_2.
#
# See docs/experiments/EXP-006a-ham10000-jepa-adam-mlp-v1.md §7 for
# the decision framework and contamination caveat.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

: "${DERMA_JEPA_HAM10000_REPO:=abdelstark/ham10000}"
: "${DERMA_JEPA_RUN_ID:=ham10000-hf-dermlip-primary-exp007-$(date -u +%Y%m%d-%H%M%S)}"
: "${HF_JOBS_FLAVOR:=a10g-large}"
: "${HF_JOBS_TIMEOUT:=12h}"
: "${HF_JOBS_DETACH:=1}"

export DERMA_JEPA_CONFIG_PATH="configs/data/ham10000_hf_mounted_exp007.yaml"
export DERMA_JEPA_INSTALL_EXTRAS="model"
export HF_JOBS_VOLUME="hf://datasets/${DERMA_JEPA_HAM10000_REPO}:/data:ro"
export HF_JOBS_FLAVOR
export HF_JOBS_TIMEOUT
export HF_JOBS_DETACH
export DERMA_JEPA_RUN_ID

printf 'Launching DermaJEPA EXP-007 (DermLIP PanDerm / EXP-004 proxy)\n'
printf '  run_id        = %s\n' "$DERMA_JEPA_RUN_ID"
printf '  dataset mount = %s\n' "$HF_JOBS_VOLUME"
printf '  flavor        = %s\n' "$HF_JOBS_FLAVOR"
printf '  timeout       = %s\n' "$HF_JOBS_TIMEOUT"
printf '  config        = %s\n' "$DERMA_JEPA_CONFIG_PATH"
printf '\n'

exec "$SCRIPT_DIR/hf_jobs_train_bundle.sh" "$@"
