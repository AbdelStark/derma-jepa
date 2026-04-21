from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

PairLabel = Literal["stable", "changing"]
Split = Literal["train", "val", "test"]

SPLITS: tuple[Split, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestRow:
    pair_id: str
    split: Split
    source_dataset: str
    pair_label: PairLabel
    context_image_id: str
    target_image_id: str
    context_path: str
    target_path: str
    context_checksum: str
    target_checksum: str
    context_width: int
    context_height: int
    target_width: int
    target_height: int
    context_patient_id: str
    target_patient_id: str
    context_lesion_id: str
    target_lesion_id: str
    diagnosis: str
    anatomical_site: str
    preprocessing_profile: str
    augmentation_recipe_json: str
    pair_construction_reason: str

    @property
    def label_int(self) -> int:
        return 1 if self.pair_label == "changing" else 0


MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string()),
        pa.field("split", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("pair_label", pa.string()),
        pa.field("context_image_id", pa.string()),
        pa.field("target_image_id", pa.string()),
        pa.field("context_path", pa.string()),
        pa.field("target_path", pa.string()),
        pa.field("context_checksum", pa.string()),
        pa.field("target_checksum", pa.string()),
        pa.field("context_width", pa.int64()),
        pa.field("context_height", pa.int64()),
        pa.field("target_width", pa.int64()),
        pa.field("target_height", pa.int64()),
        pa.field("context_patient_id", pa.string()),
        pa.field("target_patient_id", pa.string()),
        pa.field("context_lesion_id", pa.string()),
        pa.field("target_lesion_id", pa.string()),
        pa.field("diagnosis", pa.string()),
        pa.field("anatomical_site", pa.string()),
        pa.field("preprocessing_profile", pa.string()),
        pa.field("augmentation_recipe_json", pa.string()),
        pa.field("pair_construction_reason", pa.string()),
    ]
)


def manifest_to_table(rows: list[ManifestRow]) -> pa.Table:
    return pa.Table.from_pylist([asdict(row) for row in rows], schema=MANIFEST_SCHEMA)


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = manifest_to_table(rows)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def read_manifest(path: Path) -> list[ManifestRow]:
    table = pq.read_table(path, schema=MANIFEST_SCHEMA)  # type: ignore[no-untyped-call]
    return [
        ManifestRow(**_typed_manifest_record(record))
        for record in table.to_pylist()
    ]


def validate_manifest(rows: list[ManifestRow]) -> None:
    if not rows:
        msg = "manifest is empty"
        raise ValueError(msg)

    pair_ids = [row.pair_id for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        msg = "manifest contains duplicate pair_id values"
        raise ValueError(msg)

    seen_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    patients_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        if row.split not in SPLITS:
            msg = f"invalid split {row.split!r}"
            raise ValueError(msg)
        if row.pair_label not in ("stable", "changing"):
            msg = f"invalid pair label {row.pair_label!r}"
            raise ValueError(msg)
        if row.preprocessing_profile == "":
            msg = f"missing preprocessing profile for {row.pair_id}"
            raise ValueError(msg)
        if row.pair_label == "stable" and row.context_lesion_id != row.target_lesion_id:
            msg = f"stable pair crosses lesion IDs: {row.pair_id}"
            raise ValueError(msg)
        if (
            row.pair_label == "changing"
            and row.context_lesion_id == row.target_lesion_id
        ):
            msg = f"changing pair reuses lesion ID: {row.pair_id}"
            raise ValueError(msg)
        seen_by_split[row.split].update([row.context_lesion_id, row.target_lesion_id])
        patients_by_split[row.split].update(
            [row.context_patient_id, row.target_patient_id]
        )

    _assert_disjoint("lesion", seen_by_split)
    _assert_disjoint("patient", patients_by_split)

    for split in SPLITS:
        labels = {row.pair_label for row in rows if row.split == split}
        if labels != {"stable", "changing"}:
            msg = f"split {split} must contain stable and changing pairs"
            raise ValueError(msg)


def _assert_disjoint(name: str, values_by_split: dict[str, set[str]]) -> None:
    for left, right in combinations(SPLITS, 2):
        overlap = values_by_split[left] & values_by_split[right]
        if overlap:
            msg = f"{name} leakage across {left}/{right}: {sorted(overlap)}"
            raise ValueError(msg)


def _typed_manifest_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": str(record["pair_id"]),
        "split": record["split"],
        "source_dataset": str(record["source_dataset"]),
        "pair_label": record["pair_label"],
        "context_image_id": str(record["context_image_id"]),
        "target_image_id": str(record["target_image_id"]),
        "context_path": str(record["context_path"]),
        "target_path": str(record["target_path"]),
        "context_checksum": str(record["context_checksum"]),
        "target_checksum": str(record["target_checksum"]),
        "context_width": int(record["context_width"]),
        "context_height": int(record["context_height"]),
        "target_width": int(record["target_width"]),
        "target_height": int(record["target_height"]),
        "context_patient_id": str(record["context_patient_id"]),
        "target_patient_id": str(record["target_patient_id"]),
        "context_lesion_id": str(record["context_lesion_id"]),
        "target_lesion_id": str(record["target_lesion_id"]),
        "diagnosis": str(record["diagnosis"]),
        "anatomical_site": str(record["anatomical_site"]),
        "preprocessing_profile": str(record["preprocessing_profile"]),
        "augmentation_recipe_json": str(record["augmentation_recipe_json"]),
        "pair_construction_reason": str(record["pair_construction_reason"]),
    }
