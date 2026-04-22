from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from derma_jepa.config import EmbeddingModelConfig, PipelineConfig
from derma_jepa.contracts import ManifestRow, read_manifest
from derma_jepa.preprocessing import load_preprocessed_rgb
from derma_jepa.run import append_log, prepare_run_dir, write_json


def export_embeddings(config: PipelineConfig) -> Path:
    """Export configured image embeddings for the manifest in a completed run."""
    run_dir = prepare_run_dir(config)
    rows = read_manifest(run_dir / "manifest_all.parquet")
    image_records = _unique_images(rows)
    if not config.embedding_models:
        msg = "config.embeddings.models must contain at least one model"
        raise ValueError(msg)

    index_records: list[dict[str, str | int]] = []
    for model in config.embedding_models:
        artifact_path, metadata_path, dimension = _export_model_embeddings(
            config,
            model,
            image_records,
        )
        index_records.append(
            {
                "model_id": model.model_id,
                "kind": model.kind,
                "model_name": model.model_name or model.model_id,
                "artifact_path": str(artifact_path),
                "metadata_path": str(metadata_path),
                "dimension": dimension,
                "image_count": len(image_records),
            }
        )

    index_path = run_dir / "artifacts" / "embeddings" / "embedding_index.json"
    write_json(
        index_path,
        {
            "run_id": config.run_id,
            "tier": config.tier,
            "models": index_records,
            "clinical_boundary": (
                "research embedding export only; not diagnostic and not medical advice"
            ),
        },
    )
    append_log(
        run_dir,
        "embed.log",
        (
            f"exported {len(index_records)} embedding model(s) for "
            f"{len(image_records)} images"
        ),
    )
    return index_path


def export_fixture_embeddings(config: PipelineConfig) -> Path:
    """Backward-compatible fixture embedding entrypoint used by the fixture pipeline."""
    export_embeddings(config)
    return config.run_dir / "artifacts" / "embeddings" / "fixture_embeddings.npz"


def read_embedding_vectors(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        image_ids = [str(image_id) for image_id in payload["image_id"].tolist()]
        matrix = payload["vector"].astype(np.float32)
    if len(image_ids) != matrix.shape[0]:
        msg = f"Embedding artifact row mismatch: {path}"
        raise ValueError(msg)
    return {
        image_id: _l2_normalize_vector(matrix[index])
        for index, image_id in enumerate(image_ids)
    }


def _export_model_embeddings(
    config: PipelineConfig,
    model: EmbeddingModelConfig,
    image_records: dict[str, str],
) -> tuple[Path, Path, int]:
    if model.kind == "color_texture":
        matrix = _color_texture_matrix(config, image_records)
        feature_type = "color_texture_summary"
    elif model.kind == "dinov2":
        matrix = _dinov2_matrix(model, image_records)
        feature_type = "dinov2_cls_token"
    else:
        msg = f"unsupported embedding model kind: {model.kind}"
        raise ValueError(msg)

    image_ids = [image_id for image_id, _ in sorted(image_records.items())]
    paths = [path for _, path in sorted(image_records.items())]
    stem = _artifact_stem(model)
    out_path = config.run_dir / "artifacts" / "embeddings" / f"{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        image_id=np.asarray(image_ids),
        vector=matrix.astype(np.float32),
        model_id=np.asarray([model.model_id]),
    )

    metadata_path = config.run_dir / "artifacts" / "embeddings" / f"{stem}.parquet"
    metadata = pa.Table.from_pylist(
        [
            {
                "image_id": image_id,
                "path": path,
                "model_id": model.model_id,
                "model_name": model.model_name or model.model_id,
                "model_kind": model.kind,
                "feature_type": feature_type,
                "dimension": int(matrix.shape[1]),
                "preprocessing_profile": config.preprocessing.profile,
            }
            for image_id, path in zip(image_ids, paths, strict=True)
        ]
    )
    pq.write_table(metadata, metadata_path)  # type: ignore[no-untyped-call]
    return out_path, metadata_path, int(matrix.shape[1])


def _unique_images(rows: list[ManifestRow]) -> dict[str, str]:
    images: dict[str, str] = {}
    for row in rows:
        images[row.context_image_id] = row.context_path
        images[row.target_image_id] = row.target_path
    return dict(sorted(images.items()))


def _color_texture_matrix(
    config: PipelineConfig,
    image_records: dict[str, str],
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for _, path in sorted(image_records.items()):
        arr = load_preprocessed_rgb(Path(path), config.preprocessing.image_size)
        vectors.append(_feature_vector(arr))
    return _l2_normalize_matrix(np.stack(vectors).astype(np.float32))


def _dinov2_matrix(
    model_config: EmbeddingModelConfig,
    image_records: dict[str, str],
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        msg = (
            "DINOv2 embedding export requires optional model dependencies. "
            "Install them with `uv sync --extra model` before running "
            "`derma-jepa embed` for a DINOv2 config."
        )
        raise RuntimeError(msg) from exc

    device = _resolve_device(torch, model_config.device)
    processor = AutoImageProcessor.from_pretrained(model_config.model_name)
    model = AutoModel.from_pretrained(model_config.model_name)
    model.to(device)
    model.eval()

    all_vectors: list[np.ndarray] = []
    paths = [path for _, path in sorted(image_records.items())]
    with torch.no_grad():
        for start in range(0, len(paths), model_config.batch_size):
            batch_paths = paths[start : start + model_config.batch_size]
            images = [_load_pil_rgb(path) for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            vectors = _output_vectors(outputs)
            all_vectors.append(vectors.detach().cpu().numpy().astype(np.float32))
    return _l2_normalize_matrix(np.concatenate(all_vectors, axis=0))


def _load_pil_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def _output_vectors(outputs: Any) -> Any:
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        msg = "DINOv2 model output did not contain pooler_output or last_hidden_state"
        raise RuntimeError(msg)
    return hidden[:, 0]


def _feature_vector(arr: np.ndarray) -> np.ndarray:
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(
            arr[:, :, channel],
            bins=8,
            range=(0.0, 1.0),
            density=True,
        )
        histograms.append(hist.astype(np.float32))
    gray = arr.mean(axis=2)
    quantiles = np.quantile(gray, [0.05, 0.25, 0.5, 0.75, 0.95])
    coarse_grid = _coarse_grid_features(arr, cells=8)
    contrast_grid = _contrast_grid_features(arr, cells=12)
    foreground = _foreground_features(arr)
    return np.concatenate(
        [
            means,
            stds,
            *histograms,
            quantiles,
            coarse_grid,
            contrast_grid * 8.0,
            foreground * 4.0,
        ]
    ).astype(np.float32)


def _coarse_grid_features(arr: np.ndarray, *, cells: int) -> np.ndarray:
    rows = np.array_split(np.arange(arr.shape[0]), cells)
    cols = np.array_split(np.arange(arr.shape[1]), cells)
    features: list[np.ndarray] = []
    for row_indices in rows:
        for col_indices in cols:
            patch = arr[np.ix_(row_indices, col_indices)]
            features.append(patch.mean(axis=(0, 1)))
    return np.concatenate(features).astype(np.float32)


def _contrast_grid_features(arr: np.ndarray, *, cells: int) -> np.ndarray:
    background = _background_color(arr)
    distance = np.linalg.norm(arr - background, axis=2)
    rows = np.array_split(np.arange(distance.shape[0]), cells)
    cols = np.array_split(np.arange(distance.shape[1]), cells)
    features: list[float] = []
    for row_indices in rows:
        for col_indices in cols:
            patch = distance[np.ix_(row_indices, col_indices)]
            features.append(float(patch.mean()))
    return np.asarray(features, dtype=np.float32)


def _foreground_features(arr: np.ndarray) -> np.ndarray:
    background = _background_color(arr)
    distance = np.linalg.norm(arr - background, axis=2)
    threshold = max(0.08, float(np.quantile(distance, 0.75)))
    mask = distance > threshold
    if not bool(mask.any()):
        return np.zeros(15, dtype=np.float32)

    y_indices, x_indices = np.nonzero(mask)
    height, width = arr.shape[:2]
    x_norm = x_indices.astype(np.float32) / max(1, width - 1)
    y_norm = y_indices.astype(np.float32) / max(1, height - 1)
    foreground = arr[mask]
    bbox_width = (float(x_indices.max()) - float(x_indices.min()) + 1.0) / width
    bbox_height = (float(y_indices.max()) - float(y_indices.min()) + 1.0) / height
    return np.asarray(
        [
            float(mask.mean()),
            float(x_norm.mean()),
            float(y_norm.mean()),
            float(x_norm.std()),
            float(y_norm.std()),
            bbox_width,
            bbox_height,
            *foreground.mean(axis=0).tolist(),
            *foreground.std(axis=0).tolist(),
            float(distance[mask].mean()),
            float(distance[mask].max()),
        ],
        dtype=np.float32,
    )


def _background_color(arr: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [
            arr[0, :, :],
            arr[-1, :, :],
            arr[:, 0, :],
            arr[:, -1, :],
        ],
        axis=0,
    )
    return cast(np.ndarray, np.median(border, axis=0))


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    return cast(np.ndarray, normalized.astype(np.float32, copy=False))


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def _artifact_stem(model: EmbeddingModelConfig) -> str:
    if model.model_id == "fixture_color_texture_v1":
        return "fixture_embeddings"
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in model.model_id
    )
