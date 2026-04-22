from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_hf_jobs_entrypoint_resolves_config_without_training(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "local",
                "output_root": str(tmp_path / "unused-runs"),
                "artifact_root": str(tmp_path / "unused-artifacts"),
                "seed": 20260422,
                "fixture": {
                    "image_size": 64,
                    "lesions_per_split": 6,
                    "stable_pairs_per_split": 4,
                    "changing_pairs_per_split": 4,
                    "source_dataset": "synthetic_fixture",
                },
                "preprocessing": {
                    "profile": "fixture_64_center_crop_v1",
                    "image_size": 64,
                },
                "metrics": {
                    "bootstrap_samples": 20,
                    "ci_level": 0.95,
                    "fixed_tpr": 0.8,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/hf_jobs_train.py",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path / "job"),
            "--run-id",
            "hf-dry-run",
            "--output-root",
            str(tmp_path / "runs"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_id: hf-dry-run" in result.stdout
    assert f"run_dir: {tmp_path / 'runs' / 'hf-dry-run'}" in result.stdout
    assert "model_id: jepa_predictor_v1" in result.stdout
