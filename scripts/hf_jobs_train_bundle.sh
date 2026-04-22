#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS_FILE_DEFAULT="$SCRIPT_DIR/hf_jobs_constraints.txt"

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

PIN_WITH_FLAGS=()
load_pin_with_flags() {
  local file="$1"
  local line trimmed
  PIN_WITH_FLAGS=()
  [[ -f "$file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    trimmed="$(printf '%s' "$line" | awk '{$1=$1; print}')"
    [[ -z "$trimmed" ]] && continue
    PIN_WITH_FLAGS+=(--with "$trimmed")
  done < "$file"
}

load_env_defaults

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with: brew install hf" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing uv. Install it before building the local training bundle." >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

DERMA_JEPA_CONFIG_PATH="${DERMA_JEPA_CONFIG_PATH:-configs/train/jepa_predictor.yaml}"
DERMA_JEPA_RUN_ID="${DERMA_JEPA_RUN_ID:-hf-jepa-fixture-$(date -u +%Y%m%d-%H%M%S)}"
DERMA_JEPA_OUTPUT_ROOT="${DERMA_JEPA_OUTPUT_ROOT:-outputs/runs}"
DERMA_JEPA_ARTIFACT_ROOT="${DERMA_JEPA_ARTIFACT_ROOT:-outputs/artifacts/demo}"
DERMA_JEPA_INSTALL_EXTRAS="${DERMA_JEPA_INSTALL_EXTRAS:-}"
HF_JOBS_FLAVOR="${HF_JOBS_FLAVOR:-cpu-upgrade}"
HF_JOBS_TIMEOUT="${HF_JOBS_TIMEOUT:-2h}"
DERMA_JEPA_PINS_FILE="${DERMA_JEPA_PINS_FILE:-$CONSTRAINTS_FILE_DEFAULT}"

if [[ ! -f "$DERMA_JEPA_CONFIG_PATH" ]]; then
  echo "Config does not exist: $DERMA_JEPA_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$DERMA_JEPA_PINS_FILE" ]]; then
  echo "Pinned constraints file does not exist: $DERMA_JEPA_PINS_FILE" >&2
  exit 1
fi

uv build --wheel --out-dir "$tmpdir/dist" >/dev/null
wheel_path="$(find "$tmpdir/dist" -name 'derma_jepa-*.whl' -print -quit)"
if [[ -z "$wheel_path" ]]; then
  echo "Failed to build DermaJEPA wheel." >&2
  exit 1
fi

job_script="$tmpdir/derma_jepa_hf_job_bundle.py"
uv run python - "$wheel_path" "$DERMA_JEPA_CONFIG_PATH" "$DERMA_JEPA_PINS_FILE" "$job_script" <<'PY'
from __future__ import annotations

import base64
import sys
from pathlib import Path

wheel_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
constraints_path = Path(sys.argv[3])
job_script = Path(sys.argv[4])
wheel_name = wheel_path.name
wheel_b64 = base64.b64encode(wheel_path.read_bytes()).decode("ascii")
config_b64 = base64.b64encode(config_path.read_bytes()).decode("ascii")
constraints_b64 = base64.b64encode(constraints_path.read_bytes()).decode("ascii")

job_script.write_text(
    f'''from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
from pathlib import Path

WHEEL_B64 = "{wheel_b64}"
WHEEL_NAME = "{wheel_name}"
CONFIG_B64 = "{config_b64}"
CONSTRAINTS_B64 = "{constraints_b64}"


def main() -> None:
    args = _parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    wheel_path = workdir / WHEEL_NAME
    config_path = workdir / "derma_jepa_train.yaml"
    constraints_path = workdir / "hf_jobs_constraints.txt"
    wheel_path.write_bytes(base64.b64decode(WHEEL_B64))
    config_path.write_bytes(base64.b64decode(CONFIG_B64))
    constraints_path.write_bytes(base64.b64decode(CONSTRAINTS_B64))

    _ensure_pip()
    package_spec = str(wheel_path)
    if args.install_extras:
        package_spec = f"{{package_spec}}[{{args.install_extras}}]"
    install_args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--constraint",
        str(constraints_path),
        package_spec,
    ]
    if os.environ.get("HF_OUTPUT_REPO_ID"):
        install_args.append("huggingface-hub>=1.0")
    subprocess.run(install_args, check=True)

    import yaml
    from derma_jepa.config import load_config
    from derma_jepa.training import train_jepa_predictor

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DermaJEPA config must be a YAML mapping")
    payload["run_id"] = args.run_id
    payload["output_root"] = args.output_root
    payload["artifact_root"] = args.artifact_root
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)
    if args.dry_run:
        print(f"resolved config: {{config_path}}")
        print(f"run_id: {{config.run_id}}")
        print(f"run_dir: {{config.run_dir}}")
        print(f"model_id: {{config.training.model_id}}")
        print(f"embedding_model_id: {{config.training.embedding_model_id}}")
        return

    metrics_path = train_jepa_predictor(config)
    print(f"training metrics written: {{metrics_path}}")
    if config.fixture is not None:
        from derma_jepa.benchmark import validate_fixture_run

        benchmark_path = validate_fixture_run(config.run_dir)
        print(f"benchmark report written: {{benchmark_path}}")

    upload_repo = os.environ.get("HF_OUTPUT_REPO_ID")
    if upload_repo:
        _upload_run_dir(
            run_dir=config.run_dir,
            repo_id=upload_repo,
            repo_type=os.environ.get("HF_OUTPUT_REPO_TYPE", "dataset"),
            path_in_repo=os.environ.get("HF_OUTPUT_PATH", config.run_id),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bundled DermaJEPA training job without GitHub access."
    )
    parser.add_argument("--workdir", default="hf-job")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--install-extras", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ensure_pip() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)


def _upload_run_dir(
    *,
    run_dir: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
) -> None:
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(run_dir),
        path_in_repo=path_in_repo,
    )
    print(f"uploaded run directory to hf://{{repo_type}}s/{{repo_id}}/{{path_in_repo}}")


if __name__ == "__main__":
    main()
''',
    encoding="utf-8",
)
PY

cmd=(
  hf jobs uv run
  --flavor "$HF_JOBS_FLAVOR"
  --timeout "$HF_JOBS_TIMEOUT"
  --with "pip"
  --label "project=derma-jepa"
  --label "task=jepa-train-bundle"
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
  cmd+=(
    --secrets HF_TOKEN
    --env "HF_OUTPUT_REPO_ID=$HF_OUTPUT_REPO_ID"
    --env "HF_OUTPUT_REPO_TYPE=${HF_OUTPUT_REPO_TYPE:-dataset}"
    --env "HF_OUTPUT_PATH=${HF_OUTPUT_PATH:-$DERMA_JEPA_RUN_ID}"
  )
fi

cmd+=(
  "$job_script"
  --run-id "$DERMA_JEPA_RUN_ID"
  --output-root "$DERMA_JEPA_OUTPUT_ROOT"
  --artifact-root "$DERMA_JEPA_ARTIFACT_ROOT"
)

if [[ -n "$DERMA_JEPA_INSTALL_EXTRAS" ]]; then
  cmd+=(--install-extras "$DERMA_JEPA_INSTALL_EXTRAS")
fi

if [[ "${HF_JOBS_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
