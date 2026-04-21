from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FixtureConfig:
    image_size: int
    lesions_per_split: int
    stable_pairs_per_split: int
    changing_pairs_per_split: int
    source_dataset: str


@dataclass(frozen=True)
class PreprocessingConfig:
    profile: str
    image_size: int


@dataclass(frozen=True)
class MetricsConfig:
    bootstrap_samples: int
    ci_level: float
    fixed_tpr: float


@dataclass(frozen=True)
class PipelineConfig:
    run_id: str
    output_root: Path
    artifact_root: Path
    seed: int
    fixture: FixtureConfig
    preprocessing: PreprocessingConfig
    metrics: MetricsConfig
    source_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def demo_dir(self) -> Path:
        return self.artifact_root / self.run_id


def load_config(path: Path) -> PipelineConfig:
    data = _read_yaml(path)
    config = parse_config(data)
    return PipelineConfig(
        run_id=config.run_id,
        output_root=config.output_root,
        artifact_root=config.artifact_root,
        seed=config.seed,
        fixture=config.fixture,
        preprocessing=config.preprocessing,
        metrics=config.metrics,
        source_path=path,
    )


def parse_config(data: dict[str, Any]) -> PipelineConfig:
    fixture = _mapping(data, "fixture")
    preprocessing = _mapping(data, "preprocessing")
    metrics = _mapping(data, "metrics")

    parsed = PipelineConfig(
        run_id=_str(data, "run_id"),
        output_root=Path(_str(data, "output_root")),
        artifact_root=Path(_str(data, "artifact_root")),
        seed=_int(data, "seed"),
        fixture=FixtureConfig(
            image_size=_int(fixture, "image_size"),
            lesions_per_split=_int(fixture, "lesions_per_split"),
            stable_pairs_per_split=_int(fixture, "stable_pairs_per_split"),
            changing_pairs_per_split=_int(fixture, "changing_pairs_per_split"),
            source_dataset=_str(fixture, "source_dataset"),
        ),
        preprocessing=PreprocessingConfig(
            profile=_str(preprocessing, "profile"),
            image_size=_int(preprocessing, "image_size"),
        ),
        metrics=MetricsConfig(
            bootstrap_samples=_int(metrics, "bootstrap_samples"),
            ci_level=_float(metrics, "ci_level"),
            fixed_tpr=_float(metrics, "fixed_tpr"),
        ),
    )
    validate_config(parsed)
    return parsed


def validate_config(config: PipelineConfig) -> None:
    if config.fixture.lesions_per_split < 4:
        msg = "fixture.lesions_per_split must be at least 4"
        raise ValueError(msg)
    if config.fixture.stable_pairs_per_split < 1:
        msg = "fixture.stable_pairs_per_split must be positive"
        raise ValueError(msg)
    if config.fixture.changing_pairs_per_split < 1:
        msg = "fixture.changing_pairs_per_split must be positive"
        raise ValueError(msg)
    if config.preprocessing.image_size < 32:
        msg = "preprocessing.image_size must be at least 32"
        raise ValueError(msg)
    if not 0.5 <= config.metrics.ci_level < 1:
        msg = "metrics.ci_level must be in [0.5, 1.0)"
        raise ValueError(msg)
    if not 0 < config.metrics.fixed_tpr < 1:
        msg = "metrics.fixed_tpr must be in (0, 1)"
        raise ValueError(msg)


def write_resolved_config(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_yaml(config), encoding="utf-8")


def to_yaml(config: PipelineConfig) -> str:
    payload: dict[str, Any] = {
        "run_id": config.run_id,
        "output_root": str(config.output_root),
        "artifact_root": str(config.artifact_root),
        "seed": config.seed,
        "fixture": {
            "image_size": config.fixture.image_size,
            "lesions_per_split": config.fixture.lesions_per_split,
            "stable_pairs_per_split": config.fixture.stable_pairs_per_split,
            "changing_pairs_per_split": config.fixture.changing_pairs_per_split,
            "source_dataset": config.fixture.source_dataset,
        },
        "preprocessing": {
            "profile": config.preprocessing.profile,
            "image_size": config.preprocessing.image_size,
        },
        "metrics": {
            "bootstrap_samples": config.metrics.bootstrap_samples,
            "ci_level": config.metrics.ci_level,
            "fixed_tpr": config.metrics.fixed_tpr,
        },
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected mapping in config: {path}"
        raise ValueError(msg)
    return payload


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        msg = f"Expected mapping for {key}"
        raise ValueError(msg)
    return value


def _str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Expected non-empty string for {key}"
        raise ValueError(msg)
    return value


def _int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        msg = f"Expected integer for {key}"
        raise ValueError(msg)
    return value


def _float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        msg = f"Expected number for {key}"
        raise ValueError(msg)
    return float(value)
