from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from derma_jepa.config import (
    EmbeddingModelConfig,
    FixtureConfig,
    MetricsConfig,
    PipelineConfig,
    PreprocessingConfig,
    PublicDatasetConfig,
    SplitFractions,
    TrainingConfig,
    load_config,
    parse_config,
)
from derma_jepa.contracts import ManifestRow


def _stub_config(
    tmp_path: Path,
    *,
    predictor: str,
    optimizer: str,
) -> PipelineConfig:
    return PipelineConfig(
        run_id="predictor-unit",
        output_root=tmp_path / "runs",
        artifact_root=tmp_path / "artifacts" / "demo",
        seed=20260422,
        fixture=FixtureConfig(
            image_size=32,
            lesions_per_split=4,
            stable_pairs_per_split=2,
            changing_pairs_per_split=2,
            source_dataset="synthetic_fixture",
        ),
        dataset=None,
        embedding_models=(),
        training=TrainingConfig(
            model_id="jepa_predictor_unit",
            embedding_model_id=None,
            epochs=50,
            batch_size=4,
            learning_rate=0.001,
            weight_decay=0.0001,
            predictor=predictor,
            hidden_dim=32,
            optimizer=optimizer,
        ),
        preprocessing=PreprocessingConfig(profile="fixture", image_size=32),
        metrics=MetricsConfig(bootstrap_samples=20, ci_level=0.95, fixed_tpr=0.8),
    )


def _pair(pair_id: str) -> ManifestRow:
    return ManifestRow(
        pair_id=pair_id,
        split="train",
        source_dataset="synthetic_fixture",
        pair_label="stable",
        context_image_id=f"{pair_id}_ctx",
        target_image_id=f"{pair_id}_tgt",
        context_path="",
        target_path="",
        context_checksum="",
        target_checksum="",
        context_width=1,
        context_height=1,
        target_width=1,
        target_height=1,
        context_patient_id="p",
        target_patient_id="p",
        context_lesion_id="l",
        target_lesion_id="l",
        diagnosis="nv",
        anatomical_site="back",
        preprocessing_profile="fixture",
        augmentation_recipe_json="{}",
        pair_construction_reason="unit_test",
    )


def test_adam_mlp_escapes_identity_warmstart(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from derma_jepa.training import _fit_mlp_predictor

    feature_dim = 16
    rng = np.random.default_rng(0)
    n_pairs = 32
    contexts = rng.standard_normal((n_pairs, feature_dim)).astype(np.float32)
    contexts /= np.linalg.norm(contexts, axis=1, keepdims=True)
    shift = rng.standard_normal(feature_dim).astype(np.float32) * 0.5
    targets = contexts + shift
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)

    vectors: dict[str, np.ndarray] = {}
    rows: list[ManifestRow] = []
    for i in range(n_pairs):
        pair_id = f"pair_{i:03d}"
        row = _pair(pair_id)
        vectors[row.context_image_id] = contexts[i]
        vectors[row.target_image_id] = targets[i]
        rows.append(row)

    config = _stub_config(tmp_path, predictor="mlp", optimizer="adam")
    state = _fit_mlp_predictor(config, rows, rows, vectors, feature_dim)

    assert state["kind"] == "mlp"
    # Adam with LR 1e-3 should move W2 noticeably away from zero; the
    # linear warm-start's exact-identity behaviour should break within
    # the small training budget configured here.
    w2 = state["w2"]
    assert np.any(np.abs(w2) > 1e-3), (
        "Adam should move W2 away from zero-initialisation"
    )
    final_loss = float(state["history"][-1]["train_loss"])
    first_loss = float(state["history"][0]["train_loss"])
    assert final_loss < first_loss
    assert final_loss < 0.01


def test_clip_embedding_kind_parses_and_validates(tmp_path: Path) -> None:
    config = parse_config(
        {
            "run_id": "clip-parse",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 20260422,
            "dataset": {
                "kind": "ham10000",
                "name": "ham10000",
                "metadata_csv": str(tmp_path / "m.csv"),
                "image_roots": [str(tmp_path / "images")],
                "image_extensions": ["jpg"],
                "stable_pairs_per_split": 2,
                "changing_pairs_per_split": 2,
                "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                "max_records": None,
            },
            "preprocessing": {
                "profile": "ham10000_64_center_crop_v1",
                "image_size": 64,
            },
            "metrics": {"bootstrap_samples": 20, "ci_level": 0.95, "fixed_tpr": 0.8},
            "embeddings": {
                "models": [
                    {
                        "model_id": "clip_vitb16",
                        "kind": "clip",
                        "model_name": "openai/clip-vit-base-patch16",
                        "batch_size": 4,
                        "device": "cpu",
                    }
                ]
            },
            "training": {
                "model_id": "jepa_clip_linear",
                "embedding_model_id": "clip_vitb16",
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.03,
                "weight_decay": 0.001,
            },
        }
    )
    assert config.embedding_models[0].kind == "clip"
    assert config.embedding_models[0].model_name == "openai/clip-vit-base-patch16"


def test_exp006_configs_parse_cleanly() -> None:
    a = load_config(Path("configs/data/ham10000_hf_mounted_exp006a.yaml"))
    b = load_config(Path("configs/data/ham10000_hf_mounted_exp006b.yaml"))
    assert a.training.predictor == "mlp"
    assert a.training.optimizer == "adam"
    assert a.training.learning_rate == pytest.approx(0.001)
    assert b.training.predictor == "linear"
    assert b.training.optimizer == "sgd"
    assert b.embedding_models[0].kind == "clip"
    assert b.embedding_models[0].model_name == "openai/clip-vit-base-patch16"
    assert b.training.embedding_model_id == "clip_vitb16"


def test_unknown_embedding_kind_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported embedding model kind"):
        parse_config(
            {
                "run_id": "bad-embed",
                "output_root": str(tmp_path / "runs"),
                "artifact_root": str(tmp_path / "artifacts" / "demo"),
                "seed": 20260422,
                "dataset": {
                    "kind": "ham10000",
                    "name": "ham10000",
                    "metadata_csv": str(tmp_path / "m.csv"),
                    "image_roots": [str(tmp_path / "images")],
                    "image_extensions": ["jpg"],
                    "stable_pairs_per_split": 2,
                    "changing_pairs_per_split": 2,
                    "split": {"train": 0.5, "val": 0.25, "test": 0.25},
                    "max_records": None,
                },
                "preprocessing": {
                    "profile": "x",
                    "image_size": 64,
                },
                "metrics": {
                    "bootstrap_samples": 20,
                    "ci_level": 0.95,
                    "fixed_tpr": 0.8,
                },
                "embeddings": {
                    "models": [
                        {
                            "model_id": "bogus",
                            "kind": "unknown_backbone",
                            "model_name": "foo/bar",
                            "batch_size": 4,
                            "device": "cpu",
                        }
                    ]
                },
            }
        )


# Silence unused-import lint on the typed dataclasses used by _stub_config.
_ = (
    EmbeddingModelConfig,
    PublicDatasetConfig,
    SplitFractions,
)
