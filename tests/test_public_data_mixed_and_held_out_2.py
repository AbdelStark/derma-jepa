from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from derma_jepa.config import PipelineConfig, parse_config
from derma_jepa.contracts import read_manifest
from derma_jepa.public_data import build_public_manifest


def test_strong_held_out_2_recipe_is_disjoint_from_other_families(
    tmp_path: Path,
) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=24)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong_held_out_2",
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")
    stables = [row for row in rows if row.pair_label == "stable"]
    assert stables

    held_out_2_only_keys = {
        "gamma",
        "color_temperature",
        "vignette_strength",
        "salt_pepper_fraction",
        "jpeg_quality_hq",
    }
    strong_only_keys = {
        "brightness",
        "contrast",
        "saturation",
        "rotation_degrees",
        "scale",
        "hflip",
        "noise_sigma",
    }
    held_out_only_keys = {
        "hue_shift_degrees",
        "posterize_bits",
        "sharpen_radius",
        "motion_blur_length_px",
        "erase_area_fraction",
    }
    for row in stables:
        recipe = json.loads(row.augmentation_recipe_json)
        assert recipe["severity"] == "strong_held_out_2"
        assert held_out_2_only_keys.issubset(recipe.keys())
        for key in strong_only_keys | held_out_only_keys:
            assert key not in recipe, (
                f"strong_held_out_2 recipe should not contain {key!r}"
            )


def test_mixed_family_training_rotates_by_pair_index(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=60)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong,strong_held_out",
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")
    stables = [row for row in rows if row.pair_label == "stable"]
    families_seen: Counter[str] = Counter()
    for row in stables:
        recipe = json.loads(row.augmentation_recipe_json)
        families_seen[recipe["severity"]] += 1
        assert recipe.get("mixture") == "strong,strong_held_out"
        assert row.pair_construction_reason == ("same_image_post_split_mixed_nuisance")
    assert families_seen["strong"] > 0
    assert families_seen["strong_held_out"] > 0
    # Deterministic 50/50 rotation by pair index.
    assert abs(families_seen["strong"] - families_seen["strong_held_out"]) <= 1


def test_mixed_train_eval_held_out_2_uses_only_family_3_for_val_test(
    tmp_path: Path,
) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=60)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong,strong_held_out",
        nuisance_severity_eval="strong_held_out_2",
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")

    train_severities = {
        json.loads(row.augmentation_recipe_json)["severity"]
        for row in rows
        if row.pair_label == "stable" and row.split == "train"
    }
    eval_severities = {
        json.loads(row.augmentation_recipe_json)["severity"]
        for row in rows
        if row.pair_label == "stable" and row.split in {"val", "test"}
    }
    assert train_severities == {"strong", "strong_held_out"}
    assert eval_severities == {"strong_held_out_2"}


def test_strong_held_out_2_moves_pixels(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=24)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong_held_out_2",
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")
    distances: list[float] = []
    for row in rows:
        if row.pair_label != "stable":
            continue
        source = np.asarray(
            Image.open(row.context_path).convert("RGB").resize((64, 64)),
            dtype=np.float32,
        )
        variant = np.asarray(
            Image.open(row.target_path).convert("RGB").resize((64, 64)),
            dtype=np.float32,
        )
        distances.append(float(np.sqrt(np.mean(np.square(source - variant)))))
    mean_distance = float(np.mean(distances))
    assert mean_distance > 5.0, (
        f"strong_held_out_2 variants suspiciously close to source ({mean_distance:.3f})"
    )


def _config(
    tmp_path: Path,
    metadata_csv: Path,
    image_root: Path,
    *,
    nuisance_severity: str,
    nuisance_severity_eval: str | None = None,
) -> PipelineConfig:
    dataset: dict[str, object] = {
        "kind": "ham10000",
        "name": "ham10000",
        "metadata_csv": str(metadata_csv),
        "image_roots": [str(image_root)],
        "image_extensions": ["jpg"],
        "stable_pairs_per_split": 4,
        "changing_pairs_per_split": 4,
        "split": {"train": 0.5, "val": 0.25, "test": 0.25},
        "max_records": None,
        "nuisance_severity": nuisance_severity,
        "changing_pair_policy": "fallback",
    }
    if nuisance_severity_eval is not None:
        dataset["nuisance_severity_eval"] = nuisance_severity_eval
    return parse_config(
        {
            "run_id": f"public-{nuisance_severity.replace(',', '-')}",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 20260422,
            "dataset": dataset,
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
) -> tuple[Path, Path]:
    raw_dir = tmp_path / "data" / "raw" / "ham10000"
    image_root = raw_dir / "images"
    image_root.mkdir(parents=True)
    metadata_csv = raw_dir / "HAM10000_metadata.csv"
    diagnoses = ("nv", "bkl", "akiec")
    sites = ("back", "arm", "leg", "trunk")
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
                    "dx": diagnoses[index % len(diagnoses)],
                    "dx_type": "histo",
                    "age": "60",
                    "sex": "unknown",
                    "localization": sites[index % len(sites)],
                }
            )
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
    color = (72 + 5 * (index % 5), 39 + 4 * (index % 4), 32 + 4 * (index % 6), 230)
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    draw.ellipse((cx - rx // 2, cy - ry // 2, cx, cy), fill=(55, 31, 25, 160))
    image.save(path, format="JPEG")
