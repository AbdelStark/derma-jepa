from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from derma_jepa.config import PipelineConfig, parse_config
from derma_jepa.contracts import read_manifest
from derma_jepa.public_data import build_public_manifest


def test_nuisance_severity_eval_splits_train_vs_eval(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=36)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong",
        nuisance_severity_eval="strong_held_out",
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")

    train_stables = [
        row for row in rows if row.pair_label == "stable" and row.split == "train"
    ]
    eval_stables = [
        row
        for row in rows
        if row.pair_label == "stable" and row.split in {"val", "test"}
    ]
    assert train_stables, "expected stable pairs in train"
    assert eval_stables, "expected stable pairs in val/test"

    train_severities = {
        json.loads(row.augmentation_recipe_json)["severity"]
        for row in train_stables
    }
    eval_severities = {
        json.loads(row.augmentation_recipe_json)["severity"]
        for row in eval_stables
    }
    assert train_severities == {"strong"}
    assert eval_severities == {"strong_held_out"}

    train_reasons = {row.pair_construction_reason for row in train_stables}
    eval_reasons = {row.pair_construction_reason for row in eval_stables}
    assert train_reasons == {"same_image_post_split_strong_nuisance"}
    assert eval_reasons == {"same_image_post_split_strong_held_out_nuisance"}


def test_held_out_recipe_records_disjoint_transforms(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=24)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong_held_out",
        nuisance_severity_eval=None,
    )
    build_public_manifest(config)
    rows = read_manifest(config.run_dir / "manifest_all.parquet")
    stables = [row for row in rows if row.pair_label == "stable"]
    assert stables
    for row in stables:
        recipe = json.loads(row.augmentation_recipe_json)
        assert recipe["severity"] == "strong_held_out"
        # Fields that exist only in the held-out family.
        for held_out_only_key in (
            "hue_shift_degrees",
            "posterize_bits",
            "sharpen_radius",
            "motion_blur_length_px",
            "erase_area_fraction",
        ):
            assert held_out_only_key in recipe
        # Fields that exist in the EXP-002 strong family but not here.
        for strong_only_key in (
            "brightness",
            "contrast",
            "saturation",
            "rotation_degrees",
            "scale",
            "hflip",
            "noise_sigma",
        ):
            assert strong_only_key not in recipe


def test_held_out_family_still_moves_pixels(tmp_path: Path) -> None:
    """Held-out variants should be perturbed enough to not be near-identity.

    This is a sanity pin — if someone regresses the recipe to no-op, the
    test fires. We only require that the variants are further from the
    source than near-zero (they should be comparable to strong but we
    don't require an exact ordering).
    """
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=24)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong_held_out",
        nuisance_severity_eval=None,
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
        f"held-out variants suspiciously close to source ({mean_distance:.3f})"
    )


def _config(
    tmp_path: Path,
    metadata_csv: Path,
    image_root: Path,
    *,
    nuisance_severity: str,
    nuisance_severity_eval: str | None,
) -> PipelineConfig:
    dataset: dict[str, object] = {
        "kind": "ham10000",
        "name": "ham10000",
        "metadata_csv": str(metadata_csv),
        "image_roots": [str(image_root)],
        "image_extensions": ["jpg"],
        "stable_pairs_per_split": 2,
        "changing_pairs_per_split": 2,
        "split": {"train": 0.5, "val": 0.25, "test": 0.25},
        "max_records": None,
        "nuisance_severity": nuisance_severity,
        "changing_pair_policy": "fallback",
    }
    if nuisance_severity_eval is not None:
        dataset["nuisance_severity_eval"] = nuisance_severity_eval
    return parse_config(
        {
            "run_id": f"public-{nuisance_severity}-{nuisance_severity_eval}",
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
