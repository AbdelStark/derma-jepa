from pathlib import Path

from derma_jepa.config import parse_config
from derma_jepa.pipeline import run_fixture_pipeline
from derma_jepa.run import read_json


def test_fixture_pipeline_writes_run_and_demo_artifacts(tmp_path: Path) -> None:
    config = parse_config(
        {
            "run_id": "fixture-e2e",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 20260421,
            "fixture": {
                "image_size": 64,
                "lesions_per_split": 6,
                "stable_pairs_per_split": 4,
                "changing_pairs_per_split": 4,
                "source_dataset": "synthetic_fixture",
            },
            "preprocessing": {"profile": "fixture_64_center_crop_v1", "image_size": 64},
            "metrics": {"bootstrap_samples": 40, "ci_level": 0.95, "fixed_tpr": 0.8},
        }
    )

    demo_case = run_fixture_pipeline(config)
    run_dir = config.run_dir

    assert demo_case.exists()
    assert (run_dir / "benchmark_report.json").exists()
    assert (run_dir / "artifacts" / "embeddings" / "fixture_embeddings.npz").exists()
    assert (run_dir / "artifacts" / "embeddings" / "embedding_index.json").exists()
    assert (run_dir / "artifacts" / "plots" / "baseline_score_histogram.png").exists()
    assert (run_dir / "artifacts" / "reports" / "baseline_failure_cases.json").exists()
    baseline_metrics = read_json(run_dir / "baseline_metrics.json")
    assert "embedding_cosine_fixture_color_texture_v1" in baseline_metrics["baselines"]
    assert baseline_metrics["strongest_baseline"]["auroc"] >= 0.95
    demo_payload = read_json(demo_case)
    assert demo_payload["safety_boundary"].startswith("research demo")
