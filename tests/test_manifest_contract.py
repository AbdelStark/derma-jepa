from pathlib import Path

from derma_jepa.config import parse_config
from derma_jepa.contracts import SPLITS, read_manifest, validate_manifest
from derma_jepa.fixtures import build_fixture_manifest


def test_fixture_manifest_is_balanced_and_leakage_checked(tmp_path: Path) -> None:
    config = parse_config(
        {
            "run_id": "manifest-contract",
            "output_root": str(tmp_path / "runs"),
            "artifact_root": str(tmp_path / "artifacts" / "demo"),
            "seed": 7,
            "fixture": {
                "image_size": 64,
                "lesions_per_split": 6,
                "stable_pairs_per_split": 3,
                "changing_pairs_per_split": 3,
                "source_dataset": "synthetic_fixture",
            },
            "preprocessing": {"profile": "fixture_64_center_crop_v1", "image_size": 64},
            "metrics": {"bootstrap_samples": 25, "ci_level": 0.95, "fixed_tpr": 0.8},
        }
    )

    manifest_path = build_fixture_manifest(config)
    rows = read_manifest(manifest_path)

    validate_manifest(rows)
    assert len(rows) == 18
    for split in SPLITS:
        split_rows = [row for row in rows if row.split == split]
        assert {row.pair_label for row in split_rows} == {"stable", "changing"}
        assert len(split_rows) == 6
