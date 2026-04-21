from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from derma_jepa.config import PipelineConfig
from derma_jepa.contracts import ManifestRow, read_manifest
from derma_jepa.preprocessing import load_preprocessed_rgb
from derma_jepa.run import append_log, prepare_run_dir


def export_fixture_embeddings(config: PipelineConfig) -> Path:
    run_dir = prepare_run_dir(config)
    manifest_path = run_dir / "manifest_all.parquet"
    rows = read_manifest(manifest_path)
    image_records = _unique_images(rows)

    image_ids: list[str] = []
    paths: list[str] = []
    vectors: list[np.ndarray] = []
    for image_id, path in sorted(image_records.items()):
        arr = load_preprocessed_rgb(Path(path), config.preprocessing.image_size)
        image_ids.append(image_id)
        paths.append(path)
        vectors.append(_feature_vector(arr))

    matrix = np.stack(vectors).astype(np.float32)
    out_path = run_dir / "artifacts" / "embeddings" / "fixture_embeddings.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, image_id=np.asarray(image_ids), vector=matrix)

    metadata = pa.Table.from_pylist(
        [
            {
                "image_id": image_id,
                "path": path,
                "model_id": "fixture_color_texture_v1",
                "feature_type": "color_texture_summary",
                "dimension": int(matrix.shape[1]),
                "preprocessing_profile": config.preprocessing.profile,
            }
            for image_id, path in zip(image_ids, paths, strict=True)
        ]
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        metadata,
        run_dir / "artifacts" / "embeddings" / "fixture_embeddings.parquet",
    )
    append_log(run_dir, "embed.log", f"exported {len(image_ids)} fixture embeddings")
    return out_path


def _unique_images(rows: list[ManifestRow]) -> dict[str, str]:
    images: dict[str, str] = {}
    for row in rows:
        images[row.context_image_id] = row.context_path
        images[row.target_image_id] = row.target_path
    return images


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
    return np.concatenate([means, stds, *histograms, quantiles]).astype(np.float32)
