from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import yaml

from derma_jepa.config import load_config
from derma_jepa.training import train_jepa_predictor


def main() -> None:
    args = _parse_args()
    workdir = Path(args.workdir or os.environ.get("DERMA_JEPA_WORKDIR", "hf-job"))
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = _materialize_config(args, workdir)
    config = load_config(config_path)

    if args.dry_run:
        print(f"resolved config: {config_path}")
        print(f"run_id: {config.run_id}")
        print(f"run_dir: {config.run_dir}")
        print(f"model_id: {config.training.model_id}")
        print(f"embedding_model_id: {config.training.embedding_model_id}")
        return

    metrics_path = train_jepa_predictor(config)
    print(f"training metrics written: {metrics_path}")
    if config.fixture is not None:
        from derma_jepa.benchmark import validate_fixture_run

        benchmark_path = validate_fixture_run(config.run_dir)
        print(f"benchmark report written: {benchmark_path}")

    upload_repo = args.upload_repo or os.environ.get("HF_OUTPUT_REPO_ID")
    if upload_repo:
        _upload_run_dir(
            run_dir=config.run_dir,
            repo_id=upload_repo,
            repo_type=args.upload_repo_type
            or os.environ.get("HF_OUTPUT_REPO_TYPE", "dataset"),
            path_in_repo=args.upload_path
            or os.environ.get("HF_OUTPUT_PATH", config.run_id),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DermaJEPA training inside a Hugging Face Job."
    )
    parser.add_argument("--config", type=Path, help="Local config path in the Job.")
    parser.add_argument(
        "--config-url",
        default=os.environ.get("DERMA_JEPA_CONFIG_URL"),
        help="Raw URL to a DermaJEPA YAML config.",
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("DERMA_JEPA_WORKDIR"),
        help="Job-local working directory for the resolved config.",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("DERMA_JEPA_RUN_ID"),
        help="Override the config run_id.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("DERMA_JEPA_OUTPUT_ROOT", "outputs/runs"),
        help="Override the config output_root.",
    )
    parser.add_argument(
        "--artifact-root",
        default=os.environ.get("DERMA_JEPA_ARTIFACT_ROOT", "outputs/artifacts/demo"),
        help="Override the config artifact_root.",
    )
    parser.add_argument(
        "--upload-repo",
        default=os.environ.get("HF_OUTPUT_REPO_ID"),
        help="Optional Hub repo ID for uploading the completed run directory.",
    )
    parser.add_argument(
        "--upload-repo-type",
        choices=("dataset", "model", "space"),
        default=os.environ.get("HF_OUTPUT_REPO_TYPE", "dataset"),
        help="Hub repo type for --upload-repo.",
    )
    parser.add_argument(
        "--upload-path",
        default=os.environ.get("HF_OUTPUT_PATH"),
        help="Path inside the Hub repo. Defaults to the run ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the config and print the launch plan without training.",
    )
    return parser.parse_args()


def _materialize_config(args: argparse.Namespace, workdir: Path) -> Path:
    raw = _read_config_text(args)
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        msg = "DermaJEPA config must be a YAML mapping"
        raise ValueError(msg)

    if args.run_id:
        payload["run_id"] = args.run_id
    if args.output_root:
        payload["output_root"] = args.output_root
    if args.artifact_root:
        payload["artifact_root"] = args.artifact_root

    config_path = workdir / "derma_jepa_train.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _read_config_text(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config.read_text(encoding="utf-8")
    if args.config_url:
        with urllib.request.urlopen(args.config_url, timeout=60) as response:
            return response.read().decode("utf-8")
    msg = "Provide --config, --config-url, or DERMA_JEPA_CONFIG_URL"
    raise ValueError(msg)


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
    print(f"uploaded run directory to hf://{repo_type}s/{repo_id}/{path_in_repo}")


if __name__ == "__main__":
    main()
