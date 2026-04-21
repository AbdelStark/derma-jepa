from __future__ import annotations

from pathlib import Path

from derma_jepa.baselines import evaluate_fixture_baselines
from derma_jepa.benchmark import validate_fixture_run
from derma_jepa.config import PipelineConfig
from derma_jepa.demo_export import export_demo_bundle
from derma_jepa.embeddings import export_fixture_embeddings
from derma_jepa.fixtures import build_fixture_manifest


def run_fixture_pipeline(config: PipelineConfig) -> Path:
    build_fixture_manifest(config)
    export_fixture_embeddings(config)
    evaluate_fixture_baselines(config, split="test")
    validate_fixture_run(config.run_dir)
    return export_demo_bundle(config.run_dir, config.demo_dir)
