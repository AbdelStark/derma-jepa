from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from derma_jepa.config import FixtureConfig, PipelineConfig, require_fixture_config
from derma_jepa.contracts import (
    SPLITS,
    ManifestRow,
    Split,
    validate_manifest,
    write_manifest,
)
from derma_jepa.run import append_log, prepare_run_dir, run_lock, write_json


@dataclass(frozen=True)
class LesionRecord:
    image_id: str
    lesion_id: str
    patient_id: str
    split: Split
    diagnosis: str
    anatomical_site: str
    path: Path


def build_fixture_manifest(config: PipelineConfig) -> Path:
    fixture = require_fixture_config(config)
    with run_lock(config.run_dir):
        run_dir = prepare_run_dir(config)
        rng = np.random.default_rng(config.seed)
        lesions = _create_lesions(fixture, run_dir, rng)
        rows = _create_pairs(config, fixture, run_dir, lesions, rng)
        validate_manifest(rows)

        write_manifest(rows, run_dir / "manifest_all.parquet")
        for split in SPLITS:
            write_manifest(
                [row for row in rows if row.split == split],
                run_dir / f"manifest_{split}.parquet",
            )

        audit_payload = _audit_payload(config, rows)
        write_json(run_dir / "data_audit.json", audit_payload)
        append_log(
            run_dir,
            "manifest.log",
            f"built fixture manifest with {len(rows)} pairs",
        )
        return run_dir / "manifest_all.parquet"


def audit_fixture_data(config: PipelineConfig) -> Path:
    manifest_path = build_fixture_manifest(config)
    append_log(config.run_dir, "data_audit.log", "fixture data audit passed")
    return manifest_path.parent / "data_audit.json"


def _create_lesions(
    fixture: FixtureConfig,
    run_dir: Path,
    rng: np.random.Generator,
) -> dict[Split, list[LesionRecord]]:
    lesions: dict[Split, list[LesionRecord]] = {split: [] for split in SPLITS}
    image_dir = run_dir / "fixture" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for split_index, split in enumerate(SPLITS):
        for lesion_index in range(fixture.lesions_per_split):
            global_index = split_index * fixture.lesions_per_split + lesion_index
            image_id = f"{split}_base_{lesion_index:03d}"
            lesion_id = f"{split}_lesion_{lesion_index:03d}"
            patient_id = f"{split}_patient_{lesion_index:03d}"
            diagnosis = (
                "synthetic_proxy_type_a"
                if lesion_index % 2 == 0
                else "synthetic_proxy_type_b"
            )
            anatomical_site = ("torso", "arm", "leg")[lesion_index % 3]
            path = image_dir / f"{image_id}.png"
            image = _synthetic_lesion_image(
                fixture.image_size,
                global_index,
                rng,
            )
            _save_png_atomic(image, path)
            lesions[split].append(
                LesionRecord(
                    image_id=image_id,
                    lesion_id=lesion_id,
                    patient_id=patient_id,
                    split=split,
                    diagnosis=diagnosis,
                    anatomical_site=anatomical_site,
                    path=path,
                )
            )
    return lesions


def _create_pairs(
    config: PipelineConfig,
    fixture: FixtureConfig,
    run_dir: Path,
    lesions: dict[Split, list[LesionRecord]],
    rng: np.random.Generator,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for split in SPLITS:
        split_lesions = lesions[split]
        for index in range(fixture.stable_pairs_per_split):
            lesion = split_lesions[index % len(split_lesions)]
            target_path, recipe = _write_stable_variant(
                config,
                run_dir,
                lesion,
                index,
                rng,
            )
            rows.append(
                _manifest_row(
                    config=config,
                    pair_id=f"{split}_stable_{index:03d}",
                    split=split,
                    pair_label="stable",
                    context=lesion,
                    target=lesion,
                    context_path=lesion.path,
                    target_image_id=f"{lesion.image_id}_stable_{index:03d}",
                    target_path=target_path,
                    augmentation_recipe=recipe,
                    reason="same_lesion_mild_nuisance_after_split",
                )
            )

        offset = fixture.stable_pairs_per_split
        for index in range(fixture.changing_pairs_per_split):
            context = split_lesions[(index + offset) % len(split_lesions)]
            target = split_lesions[(index + offset + 1) % len(split_lesions)]
            rows.append(
                _manifest_row(
                    config=config,
                    pair_id=f"{split}_changing_{index:03d}",
                    split=split,
                    pair_label="changing",
                    context=context,
                    target=target,
                    context_path=context.path,
                    target_image_id=target.image_id,
                    target_path=target.path,
                    augmentation_recipe={"family": "none", "severity": "none"},
                    reason="different_synthetic_lesion_matched_within_split",
                )
            )
    return rows


def _synthetic_lesion_image(
    image_size: int, index: int, rng: np.random.Generator
) -> Image.Image:
    base_skin = np.array([214, 165, 132], dtype=np.uint8)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    noise = rng.normal(0, 4, size=canvas.shape)
    canvas[:] = np.clip(base_skin + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")

    cx = int(image_size * (0.42 + 0.12 * ((index % 3) / 2)))
    cy = int(image_size * (0.43 + 0.10 * (((index + 1) % 4) / 3)))
    rx = int(image_size * (0.16 + 0.025 * (index % 4)))
    ry = int(image_size * (0.12 + 0.030 * ((index + 2) % 4)))
    lesion_color = (
        int(78 + 12 * (index % 5)),
        int(42 + 8 * ((index + 1) % 5)),
        int(34 + 9 * ((index + 2) % 5)),
        230,
    )
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=lesion_color)
    draw.ellipse(
        (cx - rx // 2, cy - ry // 3, cx + rx // 3, cy + ry // 2),
        fill=(95, 55, 44, 170),
    )
    for dot in range(8):
        angle = (dot + 1) * np.pi / 5
        px = cx + int(np.cos(angle) * rx * 0.55) + int(rng.integers(-2, 3))
        py = cy + int(np.sin(angle) * ry * 0.55) + int(rng.integers(-2, 3))
        draw.ellipse((px - 1, py - 1, px + 2, py + 2), fill=(45, 28, 24, 120))
    return image.filter(ImageFilter.GaussianBlur(radius=0.35))


def _write_stable_variant(
    config: PipelineConfig,
    run_dir: Path,
    lesion: LesionRecord,
    index: int,
    rng: np.random.Generator,
) -> tuple[Path, dict[str, object]]:
    with Image.open(lesion.path) as image:
        rgb = image.convert("RGB")
        brightness = 1.0 + float(rng.uniform(-0.05, 0.05))
        contrast = 1.0 + float(rng.uniform(-0.04, 0.04))
        variant = ImageEnhance.Brightness(rgb).enhance(brightness)
        variant = ImageEnhance.Contrast(variant).enhance(contrast)
        if index % 2 == 1:
            variant = variant.filter(ImageFilter.GaussianBlur(radius=0.25))
    target_path = (
        run_dir / "fixture" / "images" / f"{lesion.image_id}_stable_{index:03d}.png"
    )
    _save_png_atomic(variant, target_path)
    recipe = {
        "family": "brightness_contrast_blur",
        "severity": "mild",
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "blur_radius": 0.25 if index % 2 == 1 else 0.0,
    }
    return target_path, recipe


def _manifest_row(
    *,
    config: PipelineConfig,
    pair_id: str,
    split: Split,
    pair_label: str,
    context: LesionRecord,
    target: LesionRecord,
    context_path: Path,
    target_image_id: str,
    target_path: Path,
    augmentation_recipe: dict[str, object],
    reason: str,
) -> ManifestRow:
    fixture = require_fixture_config(config)
    context_size = _image_size(context_path)
    target_size = _image_size(target_path)
    return ManifestRow(
        pair_id=pair_id,
        split=split,
        source_dataset=fixture.source_dataset,
        pair_label=pair_label,  # type: ignore[arg-type]
        context_image_id=context.image_id,
        target_image_id=target_image_id,
        context_path=str(context_path),
        target_path=str(target_path),
        context_checksum=_sha256(context_path),
        target_checksum=_sha256(target_path),
        context_width=context_size[0],
        context_height=context_size[1],
        target_width=target_size[0],
        target_height=target_size[1],
        context_patient_id=context.patient_id,
        target_patient_id=target.patient_id,
        context_lesion_id=context.lesion_id,
        target_lesion_id=target.lesion_id,
        diagnosis=context.diagnosis,
        anatomical_site=context.anatomical_site,
        preprocessing_profile=config.preprocessing.profile,
        augmentation_recipe_json=json.dumps(augmentation_recipe, sort_keys=True),
        pair_construction_reason=reason,
    )


def _audit_payload(
    config: PipelineConfig, rows: list[ManifestRow]
) -> dict[str, object]:
    fixture = require_fixture_config(config)
    labels_by_split = {
        split: {
            "stable": sum(
                1 for row in rows if row.split == split and row.pair_label == "stable"
            ),
            "changing": sum(
                1 for row in rows if row.split == split and row.pair_label == "changing"
            ),
        }
        for split in SPLITS
    }
    return {
        "run_id": config.run_id,
        "source_dataset": fixture.source_dataset,
        "tier": "fixture",
        "synthetic": True,
        "pair_count": len(rows),
        "labels_by_split": labels_by_split,
        "leakage_risk_note": (
            "Fixture images are deterministic synthetic artifacts. Patient and lesion "
            "identifiers are split-local and validated as disjoint across "
            "train/val/test."
        ),
        "clinical_boundary": (
            "research demo only; not diagnostic and not validated for patient use"
        ),
    }


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(path)
