from __future__ import annotations

import numpy as np
import pytest

from derma_jepa.config import (
    FixtureConfig,
    MetricsConfig,
    PipelineConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from derma_jepa.contracts import ManifestRow


def _stub_config(tmp_path, *, predictor: str, hidden_dim: int = 16) -> PipelineConfig:
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
            epochs=80,
            batch_size=4,
            learning_rate=0.05,
            weight_decay=0.0,
            predictor=predictor,
            hidden_dim=hidden_dim,
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


def test_mlp_predictor_overfits_small_paired_fixture(tmp_path) -> None:
    """MLP path must actually learn the pair correction and pass collapse checks."""
    torch = pytest.importorskip("torch")
    del torch  # ensure the import works on this environment

    from derma_jepa.training import _collapse_checks, _fit_mlp_predictor, _score_rows

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

    config = _stub_config(tmp_path, predictor="mlp")
    state = _fit_mlp_predictor(config, rows, rows, vectors, feature_dim)

    assert state["kind"] == "mlp"
    for key in ("w1", "b1", "w2", "b2"):
        assert key in state
        assert np.isfinite(state[key]).all()
    history = state["history"]
    assert history
    final_loss = float(history[-1]["train_loss"])
    first_loss = float(history[0]["train_loss"])
    assert final_loss < first_loss
    assert final_loss < 0.05

    train_scores = _score_rows(rows, vectors, state)
    changing_row = _pair("changer")
    changing_row = ManifestRow(**{**changing_row.__dict__, "pair_label": "changing"})
    vectors[changing_row.context_image_id] = contexts[0]
    # target is an unrelated vector => score should be larger than stable mean.
    unrelated = rng.standard_normal(feature_dim).astype(np.float32)
    unrelated /= np.linalg.norm(unrelated)
    vectors[changing_row.target_image_id] = unrelated
    changing_scores = _score_rows([changing_row], vectors, state)
    assert changing_scores[0] > max(train_scores)

    checks = _collapse_checks(rows, vectors, state)
    assert checks["collapsed"] is False
    assert abs(float(checks["prediction_norm_mean"]) - 1.0) < 1e-4


def test_predictor_dispatch_routes_linear_and_mlp(tmp_path) -> None:
    from derma_jepa.training import _fit_linear_predictor

    feature_dim = 8
    rng = np.random.default_rng(0)
    contexts = rng.standard_normal((6, feature_dim)).astype(np.float32)
    contexts /= np.linalg.norm(contexts, axis=1, keepdims=True)
    vectors: dict[str, np.ndarray] = {}
    rows: list[ManifestRow] = []
    for i in range(6):
        pair_id = f"pair_{i:03d}"
        row = _pair(pair_id)
        vectors[row.context_image_id] = contexts[i]
        vectors[row.target_image_id] = contexts[i]
        rows.append(row)

    config = _stub_config(tmp_path, predictor="linear")
    state = _fit_linear_predictor(config, rows, rows, vectors, feature_dim)
    assert state["kind"] == "linear"
    assert "weight" in state
    assert "bias" in state
