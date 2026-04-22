from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from derma_jepa.config import PipelineConfig, require_public_dataset_config
from derma_jepa.contracts import (
    SPLITS,
    ManifestRow,
    Split,
    validate_manifest,
    write_manifest,
)
from derma_jepa.run import append_log, prepare_run_dir, run_lock, write_json

NORMALIZED_METADATA_SCHEMA = pa.schema(
    [
        pa.field("image_id", pa.string()),
        pa.field("lesion_id", pa.string()),
        pa.field("lesion_id_available", pa.bool_()),
        pa.field("patient_id", pa.string()),
        pa.field("patient_id_available", pa.bool_()),
        pa.field("diagnosis", pa.string()),
        pa.field("anatomical_site", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("path", pa.string()),
        pa.field("image_found", pa.bool_()),
        pa.field("checksum", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
    ]
)


@dataclass(frozen=True)
class PublicImageRecord:
    image_id: str
    lesion_id: str
    lesion_id_available: bool
    patient_id: str
    patient_id_available: bool
    diagnosis: str
    anatomical_site: str
    source_dataset: str
    path: str
    image_found: bool
    checksum: str
    width: int
    height: int


@dataclass(frozen=True)
class MetadataIssues:
    duplicate_image_ids: tuple[str, ...]
    missing_image_ids: tuple[str, ...]
    unreadable_image_ids: tuple[str, ...]
    metadata_row_count: int


def audit_public_dataset(config: PipelineConfig) -> Path:
    dataset = require_public_dataset_config(config)
    with run_lock(config.run_dir):
        run_dir = prepare_run_dir(config)
        records, issues = _load_public_records(config, require_images=False)
        _write_normalized_metadata(run_dir / "metadata_normalized.parquet", records)
        payload = _audit_payload(
            config=config,
            records=records,
            issues=issues,
            rows=None,
            split_groups=None,
        )
        write_json(run_dir / "data_audit.json", payload)
        append_log(
            run_dir,
            "data_audit.log",
            (
                f"audited {issues.metadata_row_count} {dataset.name} metadata rows; "
                f"{len(issues.missing_image_ids)} images missing"
            ),
        )
        return run_dir / "data_audit.json"


def build_public_manifest(config: PipelineConfig) -> Path:
    with run_lock(config.run_dir):
        run_dir = prepare_run_dir(config)
        records, issues = _load_public_records(config, require_images=True)
        _validate_manifest_inputs(records, issues)
        split_records, split_groups = _split_records(config, records)
        rng = np.random.default_rng(config.seed + 1009)
        rows = _create_proxy_pairs(config, run_dir, split_records, rng)
        validate_manifest(rows)

        write_manifest(rows, run_dir / "manifest_all.parquet")
        for split in SPLITS:
            write_manifest(
                [row for row in rows if row.split == split],
                run_dir / f"manifest_{split}.parquet",
            )
        _write_normalized_metadata(run_dir / "metadata_normalized.parquet", records)
        write_json(
            run_dir / "data_audit.json",
            _audit_payload(
                config=config,
                records=records,
                issues=issues,
                rows=rows,
                split_groups=split_groups,
            ),
        )
        append_log(
            run_dir,
            "manifest.log",
            f"built public proxy manifest with {len(rows)} pairs",
        )
        return run_dir / "manifest_all.parquet"


def _load_public_records(
    config: PipelineConfig, *, require_images: bool
) -> tuple[list[PublicImageRecord], MetadataIssues]:
    dataset = require_public_dataset_config(config)
    if not dataset.metadata_csv.exists():
        msg = f"Missing metadata CSV: {dataset.metadata_csv}"
        raise FileNotFoundError(msg)

    raw_rows = _read_csv_rows(dataset.metadata_csv)
    if dataset.max_records is not None:
        raw_rows = raw_rows[: dataset.max_records]

    image_id_counts = Counter(_image_id(row) for row in raw_rows)
    duplicate_image_ids = tuple(
        sorted(image_id for image_id, count in image_id_counts.items() if count > 1)
    )

    records: list[PublicImageRecord] = []
    missing_image_ids: list[str] = []
    unreadable_image_ids: list[str] = []
    for row in raw_rows:
        image_id = _image_id(row)
        lesion_id_raw = _optional_field(row, ("lesion_id", "lesion", "lesionid"))
        patient_id_raw = _optional_field(
            row,
            (
                "patient_id",
                "patient",
                "subject_id",
                "participant_id",
                "patient_identifier",
            ),
        )
        lesion_id_available = lesion_id_raw is not None
        patient_id_available = patient_id_raw is not None
        lesion_id = lesion_id_raw or f"missing_lesion::{image_id}"
        patient_id = patient_id_raw or f"missing_patient::{lesion_id}"
        diagnosis = (
            _optional_field(
                row,
                ("dx", "diagnosis", "diagnosis_label", "class", "label"),
            )
            or "unknown_diagnosis"
        )
        anatomical_site = (
            _optional_field(
                row,
                ("localization", "anatom_site_general", "anatomical_site", "site"),
            )
            or "unknown_site"
        )

        path = _find_image_path(image_id, dataset.image_roots, dataset.image_extensions)
        if path is None:
            missing_image_ids.append(image_id)
            records.append(
                PublicImageRecord(
                    image_id=image_id,
                    lesion_id=lesion_id,
                    lesion_id_available=lesion_id_available,
                    patient_id=patient_id,
                    patient_id_available=patient_id_available,
                    diagnosis=diagnosis,
                    anatomical_site=anatomical_site,
                    source_dataset=dataset.name,
                    path="",
                    image_found=False,
                    checksum="",
                    width=0,
                    height=0,
                )
            )
            continue

        try:
            width, height = _image_size(path)
            checksum = _sha256(path)
        except OSError:
            if require_images:
                msg = f"Unreadable image for {image_id}: {path}"
                raise ValueError(msg) from None
            unreadable_image_ids.append(image_id)
            width, height, checksum = 0, 0, ""
            image_found = False
        else:
            image_found = True

        records.append(
            PublicImageRecord(
                image_id=image_id,
                lesion_id=lesion_id,
                lesion_id_available=lesion_id_available,
                patient_id=patient_id,
                patient_id_available=patient_id_available,
                diagnosis=diagnosis,
                anatomical_site=anatomical_site,
                source_dataset=dataset.name,
                path=str(path),
                image_found=image_found,
                checksum=checksum,
                width=width,
                height=height,
            )
        )

    issues = MetadataIssues(
        duplicate_image_ids=duplicate_image_ids,
        missing_image_ids=tuple(missing_image_ids),
        unreadable_image_ids=tuple(unreadable_image_ids),
        metadata_row_count=len(raw_rows),
    )
    if require_images and missing_image_ids:
        sample = ", ".join(missing_image_ids[:5])
        msg = (
            f"{len(missing_image_ids)} metadata rows do not have local images; "
            f"first missing image IDs: {sample}"
        )
        raise FileNotFoundError(msg)
    return records, issues


def _validate_manifest_inputs(
    records: list[PublicImageRecord], issues: MetadataIssues
) -> None:
    if issues.duplicate_image_ids:
        sample = ", ".join(issues.duplicate_image_ids[:5])
        msg = f"metadata contains duplicate image_id values: {sample}"
        raise ValueError(msg)
    available_records = [record for record in records if record.image_found]
    if len(available_records) < 6:
        msg = "at least six available images are required to build train/val/test pairs"
        raise ValueError(msg)


def _split_records(
    config: PipelineConfig,
    records: list[PublicImageRecord],
) -> tuple[dict[Split, list[PublicImageRecord]], dict[str, Split]]:
    dataset = require_public_dataset_config(config)
    available_records = [record for record in records if record.image_found]
    groups: dict[str, list[PublicImageRecord]] = {}
    for record in sorted(available_records, key=_record_sort_key):
        group_id = _split_group_id(record)
        groups.setdefault(group_id, []).append(record)

    group_ids = sorted(groups)
    train_count, val_count, test_count = _split_counts(
        len(group_ids),
        dataset.split_fractions.train,
        dataset.split_fractions.val,
    )
    permutation = np.random.default_rng(config.seed).permutation(len(group_ids))
    shuffled_group_ids = [group_ids[int(index)] for index in permutation]
    split_groups: dict[str, Split] = {}
    cut_train = train_count
    cut_val = train_count + val_count
    for index, group_id in enumerate(shuffled_group_ids):
        if index < cut_train:
            split_groups[group_id] = "train"
        elif index < cut_val:
            split_groups[group_id] = "val"
        else:
            split_groups[group_id] = "test"
    if test_count != sum(1 for split in split_groups.values() if split == "test"):
        msg = "internal split allocation error"
        raise RuntimeError(msg)

    by_split: dict[Split, list[PublicImageRecord]] = {split: [] for split in SPLITS}
    for group_id, group_records in groups.items():
        by_split[split_groups[group_id]].extend(group_records)
    for split in SPLITS:
        by_split[split].sort(key=_record_sort_key)
        lesion_count = len({record.lesion_id for record in by_split[split]})
        if lesion_count < 2:
            msg = f"split {split} needs at least two lesion IDs for changing pairs"
            raise ValueError(msg)
        requested_pairs = max(
            dataset.stable_pairs_per_split,
            dataset.changing_pairs_per_split,
        )
        if len(by_split[split]) < requested_pairs:
            msg = (
                f"split {split} has {len(by_split[split])} available images but "
                f"{requested_pairs} pairs per label were requested"
            )
            raise ValueError(msg)
    return by_split, split_groups


def _split_counts(
    total_groups: int,
    train_fraction: float,
    val_fraction: float,
) -> tuple[int, int, int]:
    if total_groups < 3:
        msg = "at least three split groups are required"
        raise ValueError(msg)
    train_count = max(1, int(round(total_groups * train_fraction)))
    val_count = max(1, int(round(total_groups * val_fraction)))
    if train_count + val_count >= total_groups:
        train_count = max(1, total_groups - 2)
        val_count = 1
    test_count = total_groups - train_count - val_count
    if test_count < 1:
        msg = "split allocation left no test groups"
        raise ValueError(msg)
    return train_count, val_count, test_count


def _create_proxy_pairs(
    config: PipelineConfig,
    run_dir: Path,
    split_records: dict[Split, list[PublicImageRecord]],
    rng: np.random.Generator,
) -> list[ManifestRow]:
    dataset = require_public_dataset_config(config)
    rows: list[ManifestRow] = []
    for split in SPLITS:
        records = split_records[split]
        for index in range(dataset.stable_pairs_per_split):
            record = records[index]
            target_path, recipe = _write_stable_variant(
                config,
                run_dir,
                split,
                record,
                index,
                rng,
            )
            rows.append(
                _manifest_row(
                    config=config,
                    pair_id=f"{split}_public_stable_{index:04d}",
                    split=split,
                    pair_label="stable",
                    context=record,
                    target=record,
                    context_path=Path(record.path),
                    target_image_id=f"{record.image_id}_stable_{index:04d}",
                    target_path=target_path,
                    augmentation_recipe=recipe,
                    reason="same_image_post_split_mild_nuisance",
                )
            )

        offset = dataset.stable_pairs_per_split
        for index in range(dataset.changing_pairs_per_split):
            context = records[(index + offset) % len(records)]
            target, reason = _match_changing_target(context, records, index)
            rows.append(
                _manifest_row(
                    config=config,
                    pair_id=f"{split}_public_changing_{index:04d}",
                    split=split,
                    pair_label="changing",
                    context=context,
                    target=target,
                    context_path=Path(context.path),
                    target_image_id=target.image_id,
                    target_path=Path(target.path),
                    augmentation_recipe={"family": "none", "severity": "none"},
                    reason=reason,
                )
            )
    return rows


def _match_changing_target(
    context: PublicImageRecord,
    records: list[PublicImageRecord],
    index: int,
) -> tuple[PublicImageRecord, str]:
    candidate_sets = [
        (
            "different_lesion_same_patient",
            [
                record
                for record in records
                if record.lesion_id != context.lesion_id
                and context.patient_id_available
                and record.patient_id_available
                and record.patient_id == context.patient_id
            ],
        ),
        (
            "different_lesion_same_diagnosis_and_site",
            [
                record
                for record in records
                if record.lesion_id != context.lesion_id
                and record.diagnosis == context.diagnosis
                and record.anatomical_site == context.anatomical_site
            ],
        ),
        (
            "different_lesion_same_diagnosis",
            [
                record
                for record in records
                if record.lesion_id != context.lesion_id
                and record.diagnosis == context.diagnosis
            ],
        ),
        (
            "different_lesion_same_site",
            [
                record
                for record in records
                if record.lesion_id != context.lesion_id
                and record.anatomical_site == context.anatomical_site
            ],
        ),
        (
            "different_lesion_unmatched_fallback",
            [record for record in records if record.lesion_id != context.lesion_id],
        ),
    ]
    for reason, candidates in candidate_sets:
        if candidates:
            candidates.sort(key=_record_sort_key)
            return candidates[index % len(candidates)], reason
    msg = f"no changing-pair target available for {context.image_id}"
    raise ValueError(msg)


def _write_stable_variant(
    config: PipelineConfig,
    run_dir: Path,
    split: Split,
    record: PublicImageRecord,
    index: int,
    rng: np.random.Generator,
) -> tuple[Path, dict[str, object]]:
    with Image.open(record.path) as image:
        variant = image.convert("RGB")
        brightness = 1.0 + float(rng.uniform(-0.08, 0.08))
        contrast = 1.0 + float(rng.uniform(-0.06, 0.06))
        angle = float(rng.uniform(-4.0, 4.0))
        blur_radius = 0.18 if index % 3 == 1 else 0.0
        noise_sigma = 1.5 if index % 3 == 2 else 0.0
        variant = ImageOps.fit(
            variant,
            (config.preprocessing.image_size, config.preprocessing.image_size),
            method=Image.Resampling.BICUBIC,
        )
        variant = ImageEnhance.Brightness(variant).enhance(brightness)
        variant = ImageEnhance.Contrast(variant).enhance(contrast)
        variant = variant.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            fillcolor=_corner_fill(variant),
        )
        if blur_radius > 0:
            variant = variant.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        if noise_sigma > 0:
            arr = np.asarray(variant, dtype=np.float32)
            noise = rng.normal(0.0, noise_sigma, size=arr.shape)
            variant = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    target_path = (
        run_dir
        / "public"
        / "stable_variants"
        / split
        / f"{record.image_id}_stable_{index:04d}.png"
    )
    _save_png_atomic(variant, target_path)
    recipe = {
        "family": "brightness_contrast_rotation_blur_noise",
        "severity": "mild",
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "rotation_degrees": round(angle, 4),
        "blur_radius": blur_radius,
        "noise_sigma": noise_sigma,
    }
    return target_path, recipe


def _manifest_row(
    *,
    config: PipelineConfig,
    pair_id: str,
    split: Split,
    pair_label: str,
    context: PublicImageRecord,
    target: PublicImageRecord,
    context_path: Path,
    target_image_id: str,
    target_path: Path,
    augmentation_recipe: dict[str, object],
    reason: str,
) -> ManifestRow:
    target_width, target_height = _image_size(target_path)
    return ManifestRow(
        pair_id=pair_id,
        split=split,
        source_dataset=context.source_dataset,
        pair_label=pair_label,  # type: ignore[arg-type]
        context_image_id=context.image_id,
        target_image_id=target_image_id,
        context_path=str(context_path),
        target_path=str(target_path),
        context_checksum=context.checksum,
        target_checksum=_sha256(target_path),
        context_width=context.width,
        context_height=context.height,
        target_width=target_width,
        target_height=target_height,
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
    *,
    config: PipelineConfig,
    records: list[PublicImageRecord],
    issues: MetadataIssues,
    rows: list[ManifestRow] | None,
    split_groups: dict[str, Split] | None,
) -> dict[str, Any]:
    dataset = require_public_dataset_config(config)
    found_records = [record for record in records if record.image_found]
    checksum_counts = Counter(
        record.checksum for record in found_records if record.checksum
    )
    duplicate_checksums = [
        checksum for checksum, count in checksum_counts.items() if count > 1
    ]
    payload: dict[str, Any] = {
        "run_id": config.run_id,
        "tier": "public",
        "source_dataset": dataset.name,
        "metadata_csv": str(dataset.metadata_csv),
        "image_roots": [str(path) for path in dataset.image_roots],
        "metadata_rows": issues.metadata_row_count,
        "normalized_records": len(records),
        "image_availability": {
            "found": len(found_records),
            "missing": len(issues.missing_image_ids),
            "unreadable": len(issues.unreadable_image_ids),
            "missing_sample": list(issues.missing_image_ids[:10]),
            "unreadable_sample": list(issues.unreadable_image_ids[:10]),
        },
        "metadata_coverage": {
            "lesion_id_present": sum(record.lesion_id_available for record in records),
            "patient_id_present": sum(
                record.patient_id_available for record in records
            ),
            "diagnosis_present": sum(
                record.diagnosis != "unknown_diagnosis" for record in records
            ),
            "anatomical_site_present": sum(
                record.anatomical_site != "unknown_site" for record in records
            ),
        },
        "duplicate_probes": {
            "duplicate_image_ids": list(issues.duplicate_image_ids[:20]),
            "duplicate_image_id_count": len(issues.duplicate_image_ids),
            "duplicate_checksum_count": len(duplicate_checksums),
        },
        "diagnosis_counts": dict(Counter(record.diagnosis for record in records)),
        "anatomical_site_counts": dict(
            Counter(record.anatomical_site for record in records)
        ),
        "leakage_risk_note": _leakage_risk_note(records, issues),
        "clinical_boundary": (
            "research demo only; longitudinal-proxy task, not diagnostic, "
            "not medical advice, and not validated for patient use"
        ),
    }
    if split_groups is not None:
        payload["split_group_counts"] = dict(Counter(split_groups.values()))
    if rows is not None:
        payload["pair_count"] = len(rows)
        payload["labels_by_split"] = _labels_by_split(rows)
        payload["leakage_probes"] = _manifest_leakage_probes(rows)
    return payload


def _leakage_risk_note(records: list[PublicImageRecord], issues: MetadataIssues) -> str:
    lesion_coverage = sum(record.lesion_id_available for record in records)
    patient_coverage = sum(record.patient_id_available for record in records)
    notes = [
        "Splits are generated at patient ID level when patient IDs are available "
        "and otherwise at lesion ID level.",
        "Stable nuisance variants are generated only after split assignment.",
    ]
    if patient_coverage == 0:
        notes.append(
            "HAM10000-style patient identifiers are unavailable in the metadata, "
            "so the main leakage boundary is lesion ID."
        )
    if lesion_coverage < len(records):
        notes.append(
            "Some rows lack lesion IDs and use image-level fallback groups; results "
            "from those rows must be treated as exploratory."
        )
    if issues.duplicate_image_ids:
        notes.append(
            "Duplicate image IDs are present and manifest generation refuses to "
            "continue until they are resolved."
        )
    if issues.missing_image_ids:
        notes.append(
            "Some metadata rows do not have local images; audit is inspectable, but "
            "manifest generation requires complete local image availability."
        )
    return " ".join(notes)


def _manifest_leakage_probes(rows: list[ManifestRow]) -> dict[str, Any]:
    return {
        "lesion_overlap": _split_overlaps(
            rows,
            lambda row: (row.context_lesion_id, row.target_lesion_id),
        ),
        "patient_overlap": _split_overlaps(
            rows,
            lambda row: (row.context_patient_id, row.target_patient_id),
        ),
        "source_counts_by_split": {
            split: dict(
                Counter(row.source_dataset for row in rows if row.split == split)
            )
            for split in SPLITS
        },
        "diagnosis_counts_by_split": {
            split: dict(Counter(row.diagnosis for row in rows if row.split == split))
            for split in SPLITS
        },
    }


def _split_overlaps(
    rows: list[ManifestRow],
    values: Callable[[ManifestRow], tuple[str, str]],
) -> dict[str, list[str]]:
    values_by_split: dict[Split, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        values_by_split[row.split].update(values(row))
    return {
        f"{left}_{right}": sorted(values_by_split[left] & values_by_split[right])
        for left, right in combinations(SPLITS, 2)
    }


def _labels_by_split(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    return {
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


def _write_normalized_metadata(path: Path, records: list[PublicImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [asdict(record) for record in records],
        schema=NORMALIZED_METADATA_SCHEMA,
    )
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {key: value or "" for key, value in row.items() if key is not None}
            )
    if not rows:
        msg = f"metadata CSV is empty: {path}"
        raise ValueError(msg)
    return rows


def _image_id(row: Mapping[str, str]) -> str:
    value = _optional_field(row, ("image_id", "isic_id", "image", "imageid"))
    if value is None:
        msg = "metadata row is missing an image_id column"
        raise ValueError(msg)
    return value


def _optional_field(row: Mapping[str, str], names: tuple[str, ...]) -> str | None:
    normalized = {
        key.strip().lower(): value.strip() for key, value in row.items() if key.strip()
    }
    for name in names:
        value = normalized.get(name.lower())
        if value:
            return value
    return None


def _find_image_path(
    image_id: str,
    image_roots: tuple[Path, ...],
    image_extensions: tuple[str, ...],
) -> Path | None:
    image_name = Path(image_id).name
    suffix = Path(image_name).suffix
    for root in image_roots:
        if suffix:
            candidate = root / image_name
            if candidate.exists():
                return candidate
        else:
            for extension in image_extensions:
                candidate = root / f"{image_name}.{extension}"
                if candidate.exists():
                    return candidate
    return None


def _split_group_id(record: PublicImageRecord) -> str:
    if record.patient_id_available:
        return f"patient::{record.patient_id}"
    return f"lesion::{record.lesion_id}"


def _record_sort_key(record: PublicImageRecord) -> tuple[str, str, str, str]:
    return (
        record.diagnosis,
        record.anatomical_site,
        record.lesion_id,
        record.image_id,
    )


def _image_size(path: Path | str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corner_fill(image: Image.Image) -> tuple[int, int, int]:
    pixel = image.getpixel((0, 0))
    if isinstance(pixel, tuple) and len(pixel) >= 3:
        return (int(pixel[0]), int(pixel[1]), int(pixel[2]))
    if isinstance(pixel, float):
        value = int(pixel)
        return (value, value, value)
    return (0, 0, 0)


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(path)
