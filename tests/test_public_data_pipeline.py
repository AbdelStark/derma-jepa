from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from derma_jepa.baselines import evaluate_baselines
from derma_jepa.config import PipelineConfig, parse_config
from derma_jepa.contracts import SPLITS, read_manifest, validate_manifest
from derma_jepa.embeddings import export_embeddings
from derma_jepa.public_data import audit_public_dataset, build_public_manifest
from derma_jepa.run import read_json


def test_public_dataset_manifest_is_leakage_checked_and_baseline_ready(
    tmp_path: Path,
) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=24)
    config = _public_config(tmp_path, metadata_csv, image_root)

    audit_path = audit_public_dataset(config)
    audit = read_json(audit_path)

    assert audit["tier"] == "public"
    assert audit["image_availability"]["found"] == 24
    assert audit["metadata_coverage"]["lesion_id_present"] == 24
    assert (config.run_dir / "metadata_normalized.parquet").exists()

    manifest_path = build_public_manifest(config)
    rows = read_manifest(manifest_path)
    manifest_audit = read_json(config.run_dir / "data_audit.json")

    validate_manifest(rows)
    assert len(rows) == 12
    assert (config.run_dir / "artifacts" / "reports" / "gold_audit_subset.csv").exists()
    assert manifest_audit["gold_audit_subset"].endswith("gold_audit_subset.csv")
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split]
        assert {row.pair_label for row in split_rows} == {"stable", "changing"}
        assert len(split_rows) == 4
    assert set(manifest_audit["leakage_probes"]["lesion_overlap"]) == {
        "train_val",
        "train_test",
        "val_test",
    }

    embedding_index_path = export_embeddings(config)
    embedding_index = read_json(embedding_index_path)
    assert embedding_index["models"][0]["model_id"] == "public_color_texture_v1"

    baseline_path = evaluate_baselines(config, split="test")
    baseline = read_json(baseline_path)
    assert baseline["tier"] == "public"
    assert "embedding_cosine_public_color_texture_v1" in baseline["baselines"]
    assert baseline["strongest_baseline"]["name"] in {
        "pixel_l2",
        "ssim_distance",
        "embedding_cosine_public_color_texture_v1",
    }
    assert (config.run_dir / "metrics.json").exists()
    assert (
        config.run_dir / "artifacts" / "plots" / "baseline_score_histogram.png"
    ).exists()
    assert (
        config.run_dir / "artifacts" / "reports" / "baseline_failure_cases.json"
    ).exists()


def test_public_dataset_audit_reports_missing_images_before_manifest(
    tmp_path: Path,
) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(
        tmp_path,
        count=9,
        missing_image_ids={"ISIC_0000008"},
    )
    config = _public_config(tmp_path, metadata_csv, image_root)

    audit_path = audit_public_dataset(config)
    audit = read_json(audit_path)

    assert audit["image_availability"]["missing"] == 1
    assert audit["image_availability"]["missing_sample"] == ["ISIC_0000008"]
    with pytest.raises(FileNotFoundError, match="do not have local images"):
        build_public_manifest(config)


def _public_config(
    tmp_path: Path,
    metadata_csv: Path,
    image_root: Path,
) -> PipelineConfig:
    return parse_config(
        {
            "run_id": "public-proxy",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 20260422,
            "dataset": {
                "kind": "ham10000",
                "name": "ham10000",
                "metadata_csv": str(metadata_csv),
                "image_roots": [str(image_root)],
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
                        "model_id": "public_color_texture_v1",
                        "kind": "color_texture",
                        "model_name": None,
                        "batch_size": 16,
                        "device": "cpu",
                    }
                ]
            },
        }
    )


def _write_ham10000_like_fixture(
    tmp_path: Path,
    *,
    count: int,
    missing_image_ids: set[str] | None = None,
) -> tuple[Path, Path]:
    missing = missing_image_ids or set()
    raw_dir = tmp_path / "data" / "raw" / "ham10000"
    image_root = raw_dir / "images"
    image_root.mkdir(parents=True)
    metadata_csv = raw_dir / "HAM10000_metadata.csv"
    with metadata_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lesion_id",
                "image_id",
                "dx",
                "dx_type",
                "age",
                "sex",
                "localization",
            ],
        )
        writer.writeheader()
        for index in range(count):
            image_id = f"ISIC_{index:07d}"
            writer.writerow(
                {
                    "lesion_id": f"HAM_{index:07d}",
                    "image_id": image_id,
                    "dx": ("nv", "bkl", "akiec")[index % 3],
                    "dx_type": "histo",
                    "age": "60",
                    "sex": "unknown",
                    "localization": ("back", "arm", "leg", "trunk")[index % 4],
                }
            )
            if image_id not in missing:
                _write_lesion_image(image_root / f"{image_id}.jpg", index)
    return metadata_csv, image_root


def _write_lesion_image(path: Path, index: int) -> None:
    size = 80
    base = (214 + (index % 5), 168 + (index % 7), 138 + (index % 3))
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image, "RGBA")
    cx = 33 + (index % 5)
    cy = 35 + (index % 7)
    rx = 11 + (index % 4)
    ry = 9 + (index % 3)
    color = (
        72 + 5 * (index % 5),
        39 + 4 * (index % 4),
        32 + 4 * (index % 6),
        230,
    )
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    draw.ellipse((cx - rx // 2, cy - ry // 2, cx, cy), fill=(55, 31, 25, 160))
    image.save(path, format="JPEG")
