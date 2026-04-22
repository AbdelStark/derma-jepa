from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from derma_jepa.config import PipelineConfig, parse_config
from derma_jepa.contracts import read_manifest
from derma_jepa.public_data import build_public_manifest


def test_strong_nuisance_recipe_is_recorded_in_manifest(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=36)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="strong",
        changing_pair_policy="fallback",
    )
    manifest_path = build_public_manifest(config)
    rows = read_manifest(manifest_path)
    stables = [row for row in rows if row.pair_label == "stable"]
    assert stables, "expected stable pairs"
    for row in stables:
        recipe = json.loads(row.augmentation_recipe_json)
        assert recipe["severity"] == "strong"
        assert "jpeg_quality" in recipe
        assert "scale" in recipe
        assert row.pair_construction_reason == "same_image_post_split_strong_nuisance"


def test_strong_nuisance_pixels_diverge_further_from_source(tmp_path: Path) -> None:
    """Sanity check: strong variants are genuinely more perturbed than mild."""

    def _pixel_distance(severity: str) -> float:
        metadata_csv, image_root = _write_ham10000_like_fixture(
            tmp_path / severity, count=24
        )
        config = _config(
            tmp_path / severity,
            metadata_csv,
            image_root,
            nuisance_severity=severity,
            changing_pair_policy="fallback",
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
        return float(np.mean(distances))

    mild_mean = _pixel_distance("mild")
    strong_mean = _pixel_distance("strong")
    # Strong should be clearly farther. Keep the margin loose so random seeds
    # do not make the test flaky, but tight enough to catch regressions.
    assert strong_mean > mild_mean * 2.0, (
        f"strong ({strong_mean:.3f}) should be >2x mild ({mild_mean:.3f})"
    )


def test_strict_changing_pair_policy_uses_only_same_dx_site(tmp_path: Path) -> None:
    metadata_csv, image_root = _write_ham10000_like_fixture(tmp_path, count=36)
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="mild",
        changing_pair_policy="strict_same_diagnosis_site",
    )
    manifest_path = build_public_manifest(config)
    rows = read_manifest(manifest_path)
    changing = [row for row in rows if row.pair_label == "changing"]
    assert changing, "expected changing pairs"
    for row in changing:
        assert row.pair_construction_reason == (
            "different_lesion_same_diagnosis_and_site"
        )
        assert row.context_lesion_id != row.target_lesion_id


def test_strict_policy_errors_when_no_candidates(tmp_path: Path) -> None:
    # Every record has a unique (dx, site) combination, so strict policy
    # cannot find a changing partner for any anchor.
    metadata_csv, image_root = _write_ham10000_like_fixture(
        tmp_path,
        count=30,
        unique_dx_site=True,
    )
    config = _config(
        tmp_path,
        metadata_csv,
        image_root,
        nuisance_severity="mild",
        changing_pair_policy="strict_same_diagnosis_site",
    )
    with pytest.raises(ValueError, match="could not produce"):
        build_public_manifest(config)


def _config(
    tmp_path: Path,
    metadata_csv: Path,
    image_root: Path,
    *,
    nuisance_severity: str,
    changing_pair_policy: str,
) -> PipelineConfig:
    return parse_config(
        {
            "run_id": f"public-{nuisance_severity}-{changing_pair_policy}",
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
                "nuisance_severity": nuisance_severity,
                "changing_pair_policy": changing_pair_policy,
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
    unique_dx_site: bool = False,
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
            if unique_dx_site:
                # Each row gets a unique (dx, site) so strict policy cannot
                # match any changing partner.
                dx = f"dx_{index:03d}"
                site = f"site_{index:03d}"
            else:
                dx = diagnoses[index % len(diagnoses)]
                site = sites[index % len(sites)]
            writer.writerow(
                {
                    "lesion_id": f"HAM_{index:07d}",
                    "image_id": image_id,
                    "dx": dx,
                    "dx_type": "histo",
                    "age": "60",
                    "sex": "unknown",
                    "localization": site,
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
    color = (
        72 + 5 * (index % 5),
        39 + 4 * (index % 4),
        32 + 4 * (index % 6),
        230,
    )
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=color)
    draw.ellipse((cx - rx // 2, cy - ry // 2, cx, cy), fill=(55, 31, 25, 160))
    image.save(path, format="JPEG")
