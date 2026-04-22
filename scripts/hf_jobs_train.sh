#!/usr/bin/env bash
set -euo pipefail

load_env_defaults() {
  local line key value
  [[ -f .env ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < .env
}

load_env_defaults

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: brew install hf" >&2
  exit 1
fi

DERMA_JEPA_REF="${DERMA_JEPA_REF:-main}"
DERMA_JEPA_GIT_URL="${DERMA_JEPA_GIT_URL:-https://github.com/AbdelStark/derma-jepa.git}"
DERMA_JEPA_PACKAGE_SPEC="${DERMA_JEPA_PACKAGE_SPEC:-derma-jepa[model] @ git+${DERMA_JEPA_GIT_URL}@${DERMA_JEPA_REF}}"
DERMA_JEPA_CONFIG_URL="${DERMA_JEPA_CONFIG_URL:-https://raw.githubusercontent.com/AbdelStark/derma-jepa/${DERMA_JEPA_REF}/configs/train/jepa_predictor.yaml}"
DERMA_JEPA_SCRIPT_URL="${DERMA_JEPA_SCRIPT_URL:-https://raw.githubusercontent.com/AbdelStark/derma-jepa/${DERMA_JEPA_REF}/scripts/hf_jobs_train.py}"
DERMA_JEPA_RUN_ID="${DERMA_JEPA_RUN_ID:-hf-jepa-fixture-$(date -u +%Y%m%d-%H%M%S)}"
DERMA_JEPA_OUTPUT_ROOT="${DERMA_JEPA_OUTPUT_ROOT:-outputs/runs}"
DERMA_JEPA_ARTIFACT_ROOT="${DERMA_JEPA_ARTIFACT_ROOT:-outputs/artifacts/demo}"
HF_JOBS_FLAVOR="${HF_JOBS_FLAVOR:-a10g-small}"
HF_JOBS_TIMEOUT="${HF_JOBS_TIMEOUT:-2h}"

cmd=(
  hf jobs uv run
  --flavor "$HF_JOBS_FLAVOR"
  --timeout "$HF_JOBS_TIMEOUT"
  --with "$DERMA_JEPA_PACKAGE_SPEC"
  --with "huggingface-hub>=1.0"
  --label "project=derma-jepa"
  --label "task=jepa-train"
)

if [[ -n "${HF_JOBS_NAMESPACE:-}" ]]; then
  cmd+=(--namespace "$HF_JOBS_NAMESPACE")
fi

if [[ -n "${HF_JOBS_VOLUME:-}" ]]; then
  cmd+=(--volume "$HF_JOBS_VOLUME")
fi

if [[ "${HF_JOBS_DETACH:-0}" == "1" ]]; then
  cmd+=(--detach)
fi

if [[ -n "${HF_OUTPUT_REPO_ID:-}" ]]; then
  cmd+=(--secrets HF_TOKEN)
fi

cmd+=(
  "$DERMA_JEPA_SCRIPT_URL"
  --config-url "$DERMA_JEPA_CONFIG_URL"
  --run-id "$DERMA_JEPA_RUN_ID"
  --output-root "$DERMA_JEPA_OUTPUT_ROOT"
  --artifact-root "$DERMA_JEPA_ARTIFACT_ROOT"
)

if [[ -n "${HF_OUTPUT_REPO_ID:-}" ]]; then
  cmd+=(
    --upload-repo "$HF_OUTPUT_REPO_ID"
    --upload-repo-type "${HF_OUTPUT_REPO_TYPE:-dataset}"
    --upload-path "${HF_OUTPUT_PATH:-$DERMA_JEPA_RUN_ID}"
  )
fi

if [[ "${HF_JOBS_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
