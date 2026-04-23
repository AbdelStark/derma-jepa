#!/usr/bin/env bash
#
# Launch DermaJEPA EXP-003 — HAM10000 primary-tier run with held-out
# nuisance family. Uses configs/data/ham10000_hf_mounted_exp003.yaml
# which keeps the EXP-002 strict_same_diagnosis_site changing-pair
# policy, trains stable pairs under the "strong" nuisance family, and
# evaluates stable pairs under the disjoint "strong_held_out" family
# (hue / posterize / sharpen / motion-blur / random-erase / low-quality-
# JPEG).
#
# Purpose: falsification test for EXP-002. If the JEPA-style predictor's
# +0.27 AUROC delta survives an unseen nuisance family, the win is
# generalizable across augmentation distributions. If it collapses back
# toward parity with DINOv2 cosine, EXP-002's result is augmentation
# memorization and the thesis needs more work.
#
# See docs/experiments/EXP-002-ham10000-jepa-hardened-proxy-v1.md §7 for
# the framing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

: "${DERMA_JEPA_HAM10000_REPO:=abdelstark/ham10000}"
: "${DERMA_JEPA_RUN_ID:=ham10000-hf-dinov2-primary-exp003-$(date -u +%Y%m%d-%H%M%S)}"
: "${HF_JOBS_FLAVOR:=a10g-large}"
: "${HF_JOBS_TIMEOUT:=12h}"
: "${HF_JOBS_DETACH:=1}"

export DERMA_JEPA_CONFIG_PATH="configs/data/ham10000_hf_mounted_exp003.yaml"
export DERMA_JEPA_INSTALL_EXTRAS="model"
export HF_JOBS_VOLUME="hf://datasets/${DERMA_JEPA_HAM10000_REPO}:/data:ro"
export HF_JOBS_FLAVOR
export HF_JOBS_TIMEOUT
export HF_JOBS_DETACH
export DERMA_JEPA_RUN_ID

printf 'Launching DermaJEPA EXP-003 HAM10000 run (held-out nuisance)\n'
printf '  run_id        = %s\n' "$DERMA_JEPA_RUN_ID"
printf '  dataset mount = %s\n' "$HF_JOBS_VOLUME"
printf '  flavor        = %s\n' "$HF_JOBS_FLAVOR"
printf '  timeout       = %s\n' "$HF_JOBS_TIMEOUT"
printf '  config        = %s\n' "$DERMA_JEPA_CONFIG_PATH"
printf '\n'

exec "$SCRIPT_DIR/hf_jobs_train_bundle.sh" "$@"
