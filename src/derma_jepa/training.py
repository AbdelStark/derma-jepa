from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from derma_jepa.config import PipelineConfig
from derma_jepa.contracts import SPLITS, ManifestRow, read_manifest
from derma_jepa.embeddings import export_embeddings, read_embedding_vectors
from derma_jepa.metrics import binary_metric_summary
from derma_jepa.observability import log_event
from derma_jepa.run import (
    append_log,
    prepare_run_dir,
    read_json,
    run_lock,
    write_json,
)


def train_jepa_predictor(config: PipelineConfig) -> Path:
    """Train the Milestone 3 latent predictor over frozen image embeddings."""
    log_event(
        "train.jepa.start",
        run_dir=config.run_dir,
        run_id=config.run_id,
        tier=config.tier,
        model_id=config.training.model_id,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
    )
    _ensure_prerequisites(config)
    with run_lock(config.run_dir):
        run_dir = prepare_run_dir(config)
        manifest = _read_split_manifest(run_dir)
        embedding_record = _selected_embedding_record(config, run_dir)
        vectors = read_embedding_vectors(Path(str(embedding_record["artifact_path"])))
        feature_dim = _feature_dim(vectors)

        train_rows = [row for row in manifest["train"] if row.pair_label == "stable"]
        if not train_rows:
            msg = "JEPA predictor training requires stable pairs in the train split"
            raise ValueError(msg)
        val_rows = [row for row in manifest["val"] if row.pair_label == "stable"]
        if not val_rows:
            val_rows = train_rows

        start = time.perf_counter()
        state = _fit_linear_predictor(
            config,
            train_rows,
            val_rows,
            vectors,
            feature_dim,
        )
        runtime_seconds = time.perf_counter() - start

        scores_by_split = {
            split: _score_rows(rows, vectors, state["weight"], state["bias"])
            for split, rows in manifest.items()
        }
        metrics_by_split = {
            split: _metric_payload(config, rows, scores_by_split[split], seed_offset)
            for split, rows, seed_offset in (
                ("train", manifest["train"], 701),
                ("val", manifest["val"], 709),
                ("test", manifest["test"], 719),
            )
        }
        collapse_checks = _collapse_checks(
            manifest["test"],
            vectors,
            state["weight"],
            state["bias"],
        )

        checkpoint_path = run_dir / "artifacts" / "models" / "jepa_predictor.npz"
        _write_checkpoint(
            checkpoint_path,
            config,
            embedding_record,
            feature_dim,
            state["weight"],
            state["bias"],
        )
        _write_latent_artifacts(
            run_dir,
            config,
            embedding_record,
            manifest,
            vectors,
            state["weight"],
            state["bias"],
            scores_by_split,
        )
        _write_score_plot(
            run_dir / "artifacts" / "plots" / "jepa_score_histogram.png",
            _pair_score_records(manifest["test"], scores_by_split["test"]),
        )

        baseline_payload = read_json(run_dir / "baseline_metrics.json")
        metrics_payload = _run_metrics_payload(
            config,
            baseline_payload,
            metrics_by_split,
            collapse_checks,
            embedding_record,
            checkpoint_path,
            runtime_seconds,
            len(train_rows),
        )
        write_json(run_dir / "metrics.json", metrics_payload)
        _write_training_report(
            run_dir,
            config,
            embedding_record,
            state["history"],
            metrics_by_split,
            collapse_checks,
            runtime_seconds,
            len(train_rows),
        )
        _write_failure_case_report(
            run_dir / "artifacts" / "reports" / "jepa_failure_cases.json",
            config,
            manifest["test"],
            scores_by_split["test"],
        )
        _write_model_card(run_dir, metrics_payload)
        append_log(
            run_dir,
            "train.log",
            (
                f"trained {config.training.model_id} on {len(train_rows)} stable "
                f"pairs with input embedding {embedding_record['model_id']}"
            ),
        )
        log_event(
            "train.jepa.end",
            run_dir=run_dir,
            run_id=config.run_id,
            tier=config.tier,
            model_id=config.training.model_id,
            train_stable_pairs=len(train_rows),
            input_embedding=str(embedding_record["model_id"]),
            primary_auroc=float(metrics_payload["primary_score"]),
            strongest_baseline_auroc=float(
                metrics_payload["result"]["strongest_baseline"]["auroc"]
            ),
            delta_vs_baseline=float(
                metrics_payload["result"]["delta_auroc_vs_strongest_baseline"]
            ),
            collapsed=bool(metrics_payload["representation_health"]["collapsed"]),
            runtime_seconds=round(runtime_seconds, 3),
        )
        return run_dir / "metrics.json"


def _ensure_prerequisites(config: PipelineConfig) -> None:
    run_dir = config.run_dir
    if not (run_dir / "manifest_all.parquet").exists():
        if config.fixture is not None:
            from derma_jepa.fixtures import build_fixture_manifest

            build_fixture_manifest(config)
        else:
            from derma_jepa.public_data import build_public_manifest

            build_public_manifest(config)
    if not (run_dir / "artifacts" / "embeddings" / "embedding_index.json").exists():
        export_embeddings(config)
    if not (run_dir / "baseline_metrics.json").exists():
        from derma_jepa.baselines import evaluate_baselines

        evaluate_baselines(config, split="test")


def _read_split_manifest(run_dir: Path) -> dict[str, list[ManifestRow]]:
    return {
        split: read_manifest(run_dir / f"manifest_{split}.parquet")
        for split in SPLITS
    }


def _selected_embedding_record(config: PipelineConfig, run_dir: Path) -> dict[str, Any]:
    index = read_json(run_dir / "artifacts" / "embeddings" / "embedding_index.json")
    models = index.get("models")
    if not isinstance(models, list) or not models:
        msg = "embedding_index.json must contain at least one embedding model"
        raise ValueError(msg)
    requested = config.training.embedding_model_id
    for record in models:
        if not isinstance(record, dict):
            msg = "embedding_index.json contains an invalid model record"
            raise ValueError(msg)
        if requested is None or str(record["model_id"]) == requested:
            return record
    msg = f"training.embedding_model_id not found in embedding index: {requested}"
    raise ValueError(msg)


def _feature_dim(vectors: dict[str, np.ndarray]) -> int:
    first = next(iter(vectors.values()), None)
    if first is None:
        msg = "embedding artifact is empty"
        raise ValueError(msg)
    return int(first.shape[0])


def _fit_linear_predictor(
    config: PipelineConfig,
    train_rows: list[ManifestRow],
    val_rows: list[ManifestRow],
    vectors: dict[str, np.ndarray],
    feature_dim: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed + 307)
    weight = np.eye(feature_dim, dtype=np.float32)
    weight += rng.normal(0.0, 0.005, size=(feature_dim, feature_dim)).astype(np.float32)
    bias = np.zeros(feature_dim, dtype=np.float32)
    train_x, train_y = _pair_matrices(train_rows, vectors)
    val_x, val_y = _pair_matrices(val_rows, vectors)
    history: list[dict[str, float | int]] = []
    identity = np.eye(feature_dim, dtype=np.float32)

    for epoch in range(1, config.training.epochs + 1):
        order = rng.permutation(train_x.shape[0])
        for start in range(0, order.shape[0], config.training.batch_size):
            batch = order[start : start + config.training.batch_size]
            x_batch = train_x[batch]
            y_batch = train_y[batch]
            pred = x_batch @ weight + bias
            err = pred - y_batch
            grad_scale = 2.0 / float(err.size)
            grad_weight = (x_batch.T @ err) * grad_scale
            grad_weight += 2.0 * config.training.weight_decay * (weight - identity)
            grad_bias = cast(np.ndarray, err.mean(axis=0) * 2.0)
            weight -= config.training.learning_rate * grad_weight.astype(np.float32)
            bias -= config.training.learning_rate * grad_bias.astype(np.float32)

        if _should_record_epoch(epoch, config.training.epochs):
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": _loss(
                        train_x,
                        train_y,
                        weight,
                        bias,
                        config.training.weight_decay,
                        identity,
                    ),
                    "val_loss": _loss(
                        val_x,
                        val_y,
                        weight,
                        bias,
                        config.training.weight_decay,
                        identity,
                    ),
                }
            )
    return {"weight": weight, "bias": bias, "history": history}


def _should_record_epoch(epoch: int, epochs: int) -> bool:
    return epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0


def _pair_matrices(
    rows: list[ManifestRow], vectors: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for row in rows:
        x_rows.append(vectors[row.context_image_id])
        y_rows.append(vectors[row.target_image_id])
    return np.stack(x_rows).astype(np.float32), np.stack(y_rows).astype(np.float32)


def _loss(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    weight_decay: float,
    identity: np.ndarray,
) -> float:
    err = (x_matrix @ weight + bias) - y_matrix
    mse = float(np.mean(np.square(err)))
    regularizer = float(weight_decay * np.mean(np.square(weight - identity)))
    return mse + regularizer


def _score_rows(
    rows: list[ManifestRow],
    vectors: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
) -> list[float]:
    scores: list[float] = []
    for row in rows:
        predicted = _predict(vectors[row.context_image_id], weight, bias)
        target = vectors[row.target_image_id]
        scores.append(_cosine_distance(predicted, target))
    return scores


def _predict(vector: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    predicted = vector @ weight + bias
    norm = float(np.linalg.norm(predicted))
    if norm <= 1e-12:
        return cast(np.ndarray, predicted.astype(np.float32))
    return cast(np.ndarray, (predicted / norm).astype(np.float32))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    similarity = float(np.dot(left, right))
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def _metric_payload(
    config: PipelineConfig,
    rows: list[ManifestRow],
    scores: list[float],
    seed_offset: int,
) -> dict[str, Any]:
    labels = [row.label_int for row in rows]
    metrics = binary_metric_summary(
        labels,
        scores,
        bootstrap_samples=config.metrics.bootstrap_samples,
        ci_level=config.metrics.ci_level,
        fixed_tpr=config.metrics.fixed_tpr,
        seed=config.seed + seed_offset,
    )
    return {
        "metrics": metrics.to_dict(),
        "pair_scores": _pair_score_records(rows, scores),
    }


def _pair_score_records(
    rows: list[ManifestRow], scores: list[float]
) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": row.pair_id,
            "split": row.split,
            "label": row.pair_label,
            "score": score,
            "context_image_id": row.context_image_id,
            "target_image_id": row.target_image_id,
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def _collapse_checks(
    rows: list[ManifestRow],
    vectors: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, float | bool]:
    predictions = np.stack(
        [_predict(vectors[row.context_image_id], weight, bias) for row in rows]
    )
    norms = np.linalg.norm(predictions, axis=1)
    variances = np.var(predictions, axis=0)
    variance_mean = float(np.mean(variances))
    norm_mean = float(np.mean(norms))
    return {
        "prediction_norm_mean": norm_mean,
        "prediction_norm_min": float(np.min(norms)),
        "dimension_variance_mean": variance_mean,
        "dimension_variance_min": float(np.min(variances)),
        "collapsed": bool(norm_mean <= 1e-8 or variance_mean <= 1e-8),
    }


def _write_checkpoint(
    path: Path,
    config: PipelineConfig,
    embedding_record: dict[str, Any],
    feature_dim: int,
    weight: np.ndarray,
    bias: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weight=weight.astype(np.float32),
        bias=bias.astype(np.float32),
        model_id=np.asarray([config.training.model_id]),
        input_embedding_model_id=np.asarray([str(embedding_record["model_id"])]),
        feature_dim=np.asarray([feature_dim], dtype=np.int32),
    )


def _write_latent_artifacts(
    run_dir: Path,
    config: PipelineConfig,
    embedding_record: dict[str, Any],
    manifest: dict[str, list[ManifestRow]],
    vectors: dict[str, np.ndarray],
    weight: np.ndarray,
    bias: np.ndarray,
    scores_by_split: dict[str, list[float]],
) -> None:
    rows = [row for split in SPLITS for row in manifest[split]]
    scores = [score for split in SPLITS for score in scores_by_split[split]]
    predicted = np.stack(
        [_predict(vectors[row.context_image_id], weight, bias) for row in rows]
    ).astype(np.float32)
    target = np.stack([vectors[row.target_image_id] for row in rows]).astype(np.float32)
    npz_path = run_dir / "artifacts" / "embeddings" / "jepa_predictor_latents.npz"
    np.savez_compressed(
        npz_path,
        pair_id=np.asarray([row.pair_id for row in rows]),
        split=np.asarray([row.split for row in rows]),
        label=np.asarray([row.pair_label for row in rows]),
        predicted_target_vector=predicted,
        target_vector=target,
        score=np.asarray(scores, dtype=np.float32),
        model_id=np.asarray([config.training.model_id]),
        input_embedding_model_id=np.asarray([str(embedding_record["model_id"])]),
    )
    parquet_path = (
        run_dir / "artifacts" / "embeddings" / "jepa_predictor_latents.parquet"
    )
    table = pa.Table.from_pylist(
        [
            {
                "pair_id": row.pair_id,
                "split": row.split,
                "pair_label": row.pair_label,
                "score": score,
                "model_id": config.training.model_id,
                "input_embedding_model_id": str(embedding_record["model_id"]),
                "feature_type": "predicted_target_latent",
            }
            for row, score in zip(rows, scores, strict=True)
        ]
    )
    pq.write_table(table, parquet_path)  # type: ignore[no-untyped-call]


def _run_metrics_payload(
    config: PipelineConfig,
    baseline_payload: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    collapse_checks: dict[str, float | bool],
    embedding_record: dict[str, Any],
    checkpoint_path: Path,
    runtime_seconds: float,
    train_stable_pairs: int,
) -> dict[str, Any]:
    jepa_metrics = metrics_by_split["test"]["metrics"]
    strongest = baseline_payload["strongest_baseline"]
    jepa_auroc = float(jepa_metrics["auroc"])
    baseline_auroc = float(strongest["auroc"])
    delta = jepa_auroc - baseline_auroc
    return {
        "run_id": config.run_id,
        "tier": config.tier,
        "model_id": config.training.model_id,
        "primary_metric": "auroc",
        "primary_score": jepa_auroc,
        "status": "jepa_style_trained",
        "result": {
            "strongest_baseline": strongest,
            "jepa_style_model": {
                "name": config.training.model_id,
                "input_embedding_model_id": str(embedding_record["model_id"]),
                "checkpoint_path": str(checkpoint_path),
                "metrics": jepa_metrics,
            },
            "delta_auroc_vs_strongest_baseline": delta,
            "interpretation": _interpretation(
                delta,
                bool(collapse_checks["collapsed"]),
            ),
        },
        "splits": {
            split: {"jepa_style_model": metrics_by_split[split]["metrics"]}
            for split in SPLITS
        },
        "training": {
            "objective": "stable-pair latent prediction over frozen image embeddings",
            "changing_pairs_used_for_training": 0,
            "stable_pairs_used_for_training": train_stable_pairs,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "runtime_seconds": runtime_seconds,
        },
        "representation_health": collapse_checks,
        "clinical_boundary": (
            "research model artifact only; not diagnostic and not validated for "
            "patient use"
        ),
    }


def _interpretation(delta_auroc: float, collapsed: bool) -> str:
    if collapsed:
        return "JEPA-style predictor trained but failed representation collapse checks."
    if delta_auroc >= 0.05:
        return (
            "JEPA-style predictor improved over the strongest baseline on this "
            "split."
        )
    if delta_auroc >= 0:
        return (
            "JEPA-style predictor matched or slightly exceeded the strongest "
            "baseline."
        )
    return (
        "JEPA-style predictor did not beat the strongest baseline on this split; "
        "report as a legitimate negative or inconclusive result."
    )


def _write_training_report(
    run_dir: Path,
    config: PipelineConfig,
    embedding_record: dict[str, Any],
    history: list[dict[str, float | int]],
    metrics_by_split: dict[str, dict[str, Any]],
    collapse_checks: dict[str, float | bool],
    runtime_seconds: float,
    train_stable_pairs: int,
) -> None:
    write_json(
        run_dir / "artifacts" / "reports" / "jepa_training_report.json",
        {
            "run_id": config.run_id,
            "model_id": config.training.model_id,
            "input_embedding_model_id": str(embedding_record["model_id"]),
            "objective": "predict target latent from context latent for stable pairs",
            "training_pair_policy": {
                "stable_pairs_used": train_stable_pairs,
                "changing_pairs_used": 0,
                "reason": (
                    "changing pairs are held out for evaluation and are not "
                    "trained to collapse together"
                ),
            },
            "history": history,
            "metrics_by_split": {
                split: metrics_by_split[split]["metrics"] for split in SPLITS
            },
            "collapse_checks": collapse_checks,
            "runtime_seconds": runtime_seconds,
            "clinical_boundary": (
                "research training report only; not diagnostic and not medical advice"
            ),
        },
    )


def _write_failure_case_report(
    path: Path,
    config: PipelineConfig,
    rows: list[ManifestRow],
    scores: list[float],
) -> None:
    score_records = sorted(
        _pair_score_records(rows, scores), key=lambda item: float(item["score"])
    )
    rows_by_pair_id = {row.pair_id: row for row in rows}
    cases = [
        *_failure_case_items(
            "stable_high_prediction_error",
            [item for item in reversed(score_records) if item["label"] == "stable"],
            rows_by_pair_id,
        ),
        *_failure_case_items(
            "changing_low_prediction_error",
            [item for item in score_records if item["label"] == "changing"],
            rows_by_pair_id,
        ),
    ]
    write_json(
        path,
        {
            "run_id": config.run_id,
            "model_id": config.training.model_id,
            "split": "test",
            "task": "jepa_predictor_failure_case_review",
            "cases": cases,
            "clinical_boundary": (
                "research error-analysis template only; not diagnostic and not "
                "validated for patient use"
            ),
        },
    )


def _failure_case_items(
    failure_mode: str,
    candidates: list[dict[str, Any]],
    rows_by_pair_id: dict[str, ManifestRow],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in candidates[:3]:
        row = rows_by_pair_id[str(item["pair_id"])]
        cases.append(
            {
                "failure_mode": failure_mode,
                "pair_id": row.pair_id,
                "label": row.pair_label,
                "score": item["score"],
                "context_image_id": row.context_image_id,
                "target_image_id": row.target_image_id,
                "context_path": row.context_path,
                "target_path": row.target_path,
                "diagnosis": row.diagnosis,
                "anatomical_site": row.anatomical_site,
                "pair_construction_reason": row.pair_construction_reason,
                "review_question": _review_question(failure_mode),
            }
        )
    return cases


def _review_question(failure_mode: str) -> str:
    if failure_mode == "stable_high_prediction_error":
        return "Did nuisance variation dominate a pair that should remain stable?"
    return "Is this changing proxy visually too subtle or poorly matched?"


def _write_score_plot(path: Path, pair_scores: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 54
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    draw.rectangle(
        (margin, margin, margin + plot_w, margin + plot_h),
        outline=(30, 30, 30),
    )
    colors = {"stable": (48, 119, 188), "changing": (201, 76, 55)}
    selected = pair_scores[:24]
    max_score = max(float(item["score"]) for item in selected) or 1.0
    draw.text((margin, 18), "JEPA-style predictor drift scores", fill=(20, 20, 20))
    for item_index, item in enumerate(selected):
        y0 = margin + 24 + item_index * 12
        bar_w = int((float(item["score"]) / max_score) * (plot_w - 170))
        color = colors[str(item["label"])]
        draw.rectangle((margin + 120, y0, margin + 120 + bar_w, y0 + 8), fill=color)
        draw.text((margin + 6, y0 - 2), str(item["label"])[:8], fill=(30, 30, 30))
    image.save(path)


def _write_model_card(run_dir: Path, metrics_payload: dict[str, Any]) -> None:
    result = metrics_payload["result"]
    jepa = result["jepa_style_model"]
    strongest = result["strongest_baseline"]
    text = f"""# JEPA-style predictor model card

## Scope

This run trains a compact latent predictor over frozen image embeddings for the
DermaJEPA longitudinal-proxy task. It predicts target latents from context
latents for stable pairs only; changing pairs are used for evaluation.

## Evidence

- Run ID: `{metrics_payload["run_id"]}`
- Model ID: `{metrics_payload["model_id"]}`
- Input embedding model: `{jepa["input_embedding_model_id"]}`
- JEPA-style AUROC: `{jepa["metrics"]["auroc"]:.3f}`
- Strongest baseline: `{strongest["name"]}` at `{strongest["auroc"]:.3f}` AUROC
- Delta vs strongest baseline: `{result["delta_auroc_vs_strongest_baseline"]:.3f}`
- Collapse detected: `{metrics_payload["representation_health"]["collapsed"]}`

## Limitations

This is a research artifact for evaluating the DermaJEPA proxy task. It is not
diagnostic, not medical advice, and not validated for patient use. Changing
pairs are never trained to collapse together; they remain evaluation evidence.
"""
    (run_dir / "model_card.md").write_text(text, encoding="utf-8")
