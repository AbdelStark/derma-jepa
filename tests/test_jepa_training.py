from pathlib import Path

import numpy as np

from derma_jepa.config import parse_config
from derma_jepa.run import read_json
from derma_jepa.training import train_jepa_predictor


def test_jepa_predictor_trains_over_fixture_embeddings(tmp_path: Path) -> None:
    config = parse_config(
        {
            "run_id": "jepa-train",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 20260422,
            "fixture": {
                "image_size": 64,
                "lesions_per_split": 6,
                "stable_pairs_per_split": 4,
                "changing_pairs_per_split": 4,
                "source_dataset": "synthetic_fixture",
            },
            "preprocessing": {"profile": "fixture_64_center_crop_v1", "image_size": 64},
            "metrics": {"bootstrap_samples": 40, "ci_level": 0.95, "fixed_tpr": 0.8},
            "training": {
                "model_id": "jepa_predictor_fixture_test",
                "embedding_model_id": "fixture_color_texture_v1",
                "epochs": 40,
                "batch_size": 4,
                "learning_rate": 0.05,
                "weight_decay": 0.001,
            },
        }
    )

    metrics_path = train_jepa_predictor(config)
    run_dir = config.run_dir

    assert metrics_path == run_dir / "metrics.json"
    assert (run_dir / "baseline_metrics.json").exists()
    assert (run_dir / "artifacts" / "models" / "jepa_predictor.npz").exists()
    assert (
        run_dir / "artifacts" / "embeddings" / "jepa_predictor_latents.npz"
    ).exists()
    assert (
        run_dir / "artifacts" / "reports" / "jepa_training_report.json"
    ).exists()
    assert (run_dir / "artifacts" / "plots" / "jepa_score_histogram.png").exists()

    metrics = read_json(metrics_path)
    report = read_json(run_dir / "artifacts" / "reports" / "jepa_training_report.json")

    assert metrics["status"] == "jepa_style_trained"
    assert (
        metrics["result"]["jepa_style_model"]["name"]
        == "jepa_predictor_fixture_test"
    )
    assert metrics["primary_score"] >= 0.9
    assert metrics["training"]["changing_pairs_used_for_training"] == 0
    assert metrics["training"]["stable_pairs_used_for_training"] == 4
    assert metrics["representation_health"]["collapsed"] is False
    assert report["training_pair_policy"]["changing_pairs_used"] == 0
    assert len(report["history"]) >= 2

    with np.load(
        run_dir / "artifacts" / "embeddings" / "jepa_predictor_latents.npz",
        allow_pickle=False,
    ) as payload:
        assert payload["predicted_target_vector"].shape[0] == 24
        assert payload["score"].shape[0] == 24
