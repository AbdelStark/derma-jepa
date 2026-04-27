#!/usr/bin/env bash
#
# Generic seed-sweep launcher. Takes a base config and a seed, produces
# a temp config with `seed:` and `run_id:` substituted, then exec's the
# train bundle. Used to lock in seed-to-seed variance on EXP-007 and
# EXP-008 (and any future experiment) before paper draft.
#
# Usage:
#   BASE_CONFIG=configs/data/ham10000_hf_mounted_exp007.yaml \
#   SEED=1 \
#   SWEEP_TAG=dermlip-exp007 \
#     ./scripts/hf_jobs_seed_sweep.sh
#
# Optional overrides:
#   DERMA_JEPA_RUN_ID    — explicit run id (default ham10000-hf-${SWEEP_TAG}-seed-${SEED}-v1)
#   HF_JOBS_FLAVOR       — default a10g-large
#   HF_JOBS_TIMEOUT      — default 12h
#   HF_JOBS_DETACH       — default 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

: "${BASE_CONFIG:?BASE_CONFIG is required (path to base experiment yaml)}"
: "${SEED:?SEED is required (positive integer)}"
: "${SWEEP_TAG:?SWEEP_TAG is required (e.g. dermlip-exp007 or biomedclip-exp008)}"
: "${DERMA_JEPA_HAM10000_REPO:=abdelstark/ham10000}"
: "${DERMA_JEPA_RUN_ID:=ham10000-hf-${SWEEP_TAG}-seed-${SEED}-v1}"
: "${HF_JOBS_FLAVOR:=a10g-large}"
: "${HF_JOBS_TIMEOUT:=12h}"
: "${HF_JOBS_DETACH:=1}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Base config does not exist: $BASE_CONFIG" >&2
  exit 1
fi

# Generate the temp config: replace top-level `seed:` and `run_id:` lines.
tmp_config="$(mktemp -t derma_jepa_seed_sweep.XXXXXX.yaml)"
trap 'rm -f "$tmp_config"' EXIT

awk -v seed="$SEED" -v run_id="$DERMA_JEPA_RUN_ID" '
  BEGIN { sed_seed = 0; sed_run = 0 }
  /^seed:[[:space:]]/      && sed_seed == 0 { print "seed: " seed; sed_seed = 1; next }
  /^run_id:[[:space:]]/    && sed_run  == 0 { print "run_id: " run_id; sed_run  = 1; next }
  { print }
' "$BASE_CONFIG" > "$tmp_config"

if ! grep -qE "^seed: $SEED\$" "$tmp_config"; then
  echo "Failed to substitute seed in temp config" >&2
  exit 1
fi
if ! grep -qE "^run_id: $DERMA_JEPA_RUN_ID\$" "$tmp_config"; then
  echo "Failed to substitute run_id in temp config" >&2
  exit 1
fi

export DERMA_JEPA_CONFIG_PATH="$tmp_config"
export DERMA_JEPA_INSTALL_EXTRAS="model"
export HF_JOBS_VOLUME="hf://datasets/${DERMA_JEPA_HAM10000_REPO}:/data:ro"
export HF_JOBS_FLAVOR
export HF_JOBS_TIMEOUT
export HF_JOBS_DETACH
export DERMA_JEPA_RUN_ID

printf 'Launching seed-sweep run\n'
printf '  base config   = %s\n' "$BASE_CONFIG"
printf '  seed          = %s\n' "$SEED"
printf '  run_id        = %s\n' "$DERMA_JEPA_RUN_ID"
printf '  dataset mount = %s\n' "$HF_JOBS_VOLUME"
printf '  flavor        = %s\n' "$HF_JOBS_FLAVOR"
printf '  timeout       = %s\n' "$HF_JOBS_TIMEOUT"
printf '  temp config   = %s\n' "$tmp_config"
printf '\n'

exec "$SCRIPT_DIR/hf_jobs_train_bundle.sh" "$@"
