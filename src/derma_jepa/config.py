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
class SplitFractions:
    train: float
    val: float
    test: float


@dataclass(frozen=True)
class PublicDatasetConfig:
    kind: str
    name: str
    metadata_csv: Path
    image_roots: tuple[Path, ...]
    image_extensions: tuple[str, ...]
    stable_pairs_per_split: int
    changing_pairs_per_split: int
    split_fractions: SplitFractions
    max_records: int | None = None
    nuisance_severity: str = "mild"
    nuisance_severity_eval: str | None = None
    changing_pair_policy: str = "fallback"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model_id: str
    kind: str
    model_name: str | None
    batch_size: int
    device: str


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str
    embedding_model_id: str | None
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    predictor: str = "linear"
    hidden_dim: int = 512
    optimizer: str = "sgd"


@dataclass(frozen=True)
class PipelineConfig:
    run_id: str
    output_root: Path
    artifact_root: Path
    seed: int
    fixture: FixtureConfig | None
    dataset: PublicDatasetConfig | None
    embedding_models: tuple[EmbeddingModelConfig, ...]
    training: TrainingConfig
    preprocessing: PreprocessingConfig
    metrics: MetricsConfig
    source_path: Path | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def demo_dir(self) -> Path:
        return self.artifact_root / self.run_id

    @property
    def tier(self) -> str:
        if self.fixture is not None:
            return "fixture"
        return "public"


def load_config(path: Path) -> PipelineConfig:
    data = _read_yaml(path)
    config = parse_config(data)
    return PipelineConfig(
        run_id=config.run_id,
        output_root=config.output_root,
        artifact_root=config.artifact_root,
        seed=config.seed,
        fixture=config.fixture,
        dataset=config.dataset,
        embedding_models=config.embedding_models,
        training=config.training,
        preprocessing=config.preprocessing,
        metrics=config.metrics,
        source_path=path,
    )


def parse_config(data: dict[str, Any]) -> PipelineConfig:
    preprocessing = _mapping(data, "preprocessing")
    metrics = _mapping(data, "metrics")
    fixture_mapping = _optional_mapping(data, "fixture")
    dataset_mapping = _optional_mapping(data, "dataset")
    embeddings_mapping = _optional_mapping(data, "embeddings")
    training_mapping = _optional_mapping(data, "training")
    if fixture_mapping is None and dataset_mapping is None:
        msg = "Expected either fixture or dataset config"
        raise ValueError(msg)
    if fixture_mapping is not None and dataset_mapping is not None:
        msg = "fixture and dataset configs are mutually exclusive"
        raise ValueError(msg)

    parsed = PipelineConfig(
        run_id=_str(data, "run_id"),
        output_root=Path(_str(data, "output_root")),
        artifact_root=Path(_str(data, "artifact_root")),
        seed=_int(data, "seed"),
        fixture=_parse_fixture_config(fixture_mapping),
        dataset=_parse_public_dataset_config(dataset_mapping),
        embedding_models=_parse_embedding_models(
            embeddings_mapping,
            has_fixture=fixture_mapping is not None,
        ),
        training=_parse_training_config(training_mapping),
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
    if config.fixture is not None:
        if config.fixture.lesions_per_split < 4:
            msg = "fixture.lesions_per_split must be at least 4"
            raise ValueError(msg)
        if config.fixture.stable_pairs_per_split < 1:
            msg = "fixture.stable_pairs_per_split must be positive"
            raise ValueError(msg)
        if config.fixture.changing_pairs_per_split < 1:
            msg = "fixture.changing_pairs_per_split must be positive"
            raise ValueError(msg)
    if config.dataset is not None:
        if config.dataset.kind != "ham10000":
            msg = "dataset.kind must be 'ham10000'"
            raise ValueError(msg)
        if not config.dataset.image_roots:
            msg = "dataset.image_roots must contain at least one directory"
            raise ValueError(msg)
        if not config.dataset.image_extensions:
            msg = "dataset.image_extensions must contain at least one extension"
            raise ValueError(msg)
        if config.dataset.stable_pairs_per_split < 1:
            msg = "dataset.stable_pairs_per_split must be positive"
            raise ValueError(msg)
        if config.dataset.changing_pairs_per_split < 1:
            msg = "dataset.changing_pairs_per_split must be positive"
            raise ValueError(msg)
        fractions = config.dataset.split_fractions
        if min(fractions.train, fractions.val, fractions.test) <= 0:
            msg = "dataset.split fractions must be positive"
            raise ValueError(msg)
        if abs((fractions.train + fractions.val + fractions.test) - 1.0) > 1e-6:
            msg = "dataset.split fractions must sum to 1.0"
            raise ValueError(msg)
        if config.dataset.max_records is not None and config.dataset.max_records < 1:
            msg = "dataset.max_records must be positive when set"
            raise ValueError(msg)
        allowed_severities = {"mild", "strong", "strong_held_out", "strong_held_out_2"}
        for field_name, value in (
            ("nuisance_severity", config.dataset.nuisance_severity),
            ("nuisance_severity_eval", config.dataset.nuisance_severity_eval),
        ):
            if value is None:
                continue
            families = [item.strip() for item in value.split(",") if item.strip()]
            if not families:
                msg = f"dataset.{field_name} must be non-empty"
                raise ValueError(msg)
            unknown = [f for f in families if f not in allowed_severities]
            if unknown:
                msg = (
                    f"dataset.{field_name} contains unknown families "
                    f"{unknown}; allowed: {sorted(allowed_severities)}"
                )
                raise ValueError(msg)
        if config.dataset.changing_pair_policy not in {
            "fallback",
            "strict_same_diagnosis_site",
        }:
            msg = (
                "dataset.changing_pair_policy must be 'fallback' or "
                "'strict_same_diagnosis_site'"
            )
            raise ValueError(msg)
    for model in config.embedding_models:
        if model.kind not in {"color_texture", "dinov2", "clip", "open_clip"}:
            msg = f"unsupported embedding model kind: {model.kind}"
            raise ValueError(msg)
        if model.batch_size < 1:
            msg = f"embedding model {model.model_id} batch_size must be positive"
            raise ValueError(msg)
        if model.kind in {"dinov2", "clip", "open_clip"} and model.model_name is None:
            msg = f"embedding model {model.model_id} requires model_name"
            raise ValueError(msg)
    if config.training.epochs < 1:
        msg = "training.epochs must be positive"
        raise ValueError(msg)
    if config.training.batch_size < 1:
        msg = "training.batch_size must be positive"
        raise ValueError(msg)
    if config.training.learning_rate <= 0:
        msg = "training.learning_rate must be positive"
        raise ValueError(msg)
    if config.training.weight_decay < 0:
        msg = "training.weight_decay must be non-negative"
        raise ValueError(msg)
    if config.training.predictor not in {"linear", "mlp"}:
        msg = "training.predictor must be 'linear' or 'mlp'"
        raise ValueError(msg)
    if config.training.hidden_dim < 1:
        msg = "training.hidden_dim must be positive"
        raise ValueError(msg)
    if config.training.optimizer not in {"sgd", "adam"}:
        msg = "training.optimizer must be 'sgd' or 'adam'"
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
        "preprocessing": {
            "profile": config.preprocessing.profile,
            "image_size": config.preprocessing.image_size,
        },
        "metrics": {
            "bootstrap_samples": config.metrics.bootstrap_samples,
            "ci_level": config.metrics.ci_level,
            "fixed_tpr": config.metrics.fixed_tpr,
        },
        "training": {
            "model_id": config.training.model_id,
            "embedding_model_id": config.training.embedding_model_id,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "predictor": config.training.predictor,
            "hidden_dim": config.training.hidden_dim,
            "optimizer": config.training.optimizer,
        },
    }
    if config.fixture is not None:
        payload["fixture"] = {
            "image_size": config.fixture.image_size,
            "lesions_per_split": config.fixture.lesions_per_split,
            "stable_pairs_per_split": config.fixture.stable_pairs_per_split,
            "changing_pairs_per_split": config.fixture.changing_pairs_per_split,
            "source_dataset": config.fixture.source_dataset,
        }
    if config.dataset is not None:
        payload["dataset"] = {
            "kind": config.dataset.kind,
            "name": config.dataset.name,
            "metadata_csv": str(config.dataset.metadata_csv),
            "image_roots": [str(path) for path in config.dataset.image_roots],
            "image_extensions": list(config.dataset.image_extensions),
            "stable_pairs_per_split": config.dataset.stable_pairs_per_split,
            "changing_pairs_per_split": config.dataset.changing_pairs_per_split,
            "split": {
                "train": config.dataset.split_fractions.train,
                "val": config.dataset.split_fractions.val,
                "test": config.dataset.split_fractions.test,
            },
            "max_records": config.dataset.max_records,
            "nuisance_severity": config.dataset.nuisance_severity,
            "nuisance_severity_eval": config.dataset.nuisance_severity_eval,
            "changing_pair_policy": config.dataset.changing_pair_policy,
        }
    if config.embedding_models:
        payload["embeddings"] = {
            "models": [
                {
                    "model_id": model.model_id,
                    "kind": model.kind,
                    "model_name": model.model_name,
                    "batch_size": model.batch_size,
                    "device": model.device,
                }
                for model in config.embedding_models
            ]
        }
    return yaml.safe_dump(payload, sort_keys=False)


def require_fixture_config(config: PipelineConfig) -> FixtureConfig:
    if config.fixture is None:
        msg = "config does not contain a fixture section"
        raise ValueError(msg)
    return config.fixture


def require_public_dataset_config(config: PipelineConfig) -> PublicDatasetConfig:
    if config.dataset is None:
        msg = "config does not contain a dataset section"
        raise ValueError(msg)
    return config.dataset


def _parse_fixture_config(data: dict[str, Any] | None) -> FixtureConfig | None:
    if data is None:
        return None
    return FixtureConfig(
        image_size=_int(data, "image_size"),
        lesions_per_split=_int(data, "lesions_per_split"),
        stable_pairs_per_split=_int(data, "stable_pairs_per_split"),
        changing_pairs_per_split=_int(data, "changing_pairs_per_split"),
        source_dataset=_str(data, "source_dataset"),
    )


def _parse_public_dataset_config(
    data: dict[str, Any] | None,
) -> PublicDatasetConfig | None:
    if data is None:
        return None
    split = _mapping(data, "split")
    return PublicDatasetConfig(
        kind=_str(data, "kind"),
        name=_str(data, "name"),
        metadata_csv=Path(_str(data, "metadata_csv")),
        image_roots=tuple(Path(value) for value in _str_list(data, "image_roots")),
        image_extensions=tuple(
            _normalize_extension(value) for value in _str_list(data, "image_extensions")
        ),
        stable_pairs_per_split=_int(data, "stable_pairs_per_split"),
        changing_pairs_per_split=_int(data, "changing_pairs_per_split"),
        split_fractions=SplitFractions(
            train=_float(split, "train"),
            val=_float(split, "val"),
            test=_float(split, "test"),
        ),
        max_records=_optional_int(data, "max_records"),
        nuisance_severity=_optional_str(data, "nuisance_severity") or "mild",
        nuisance_severity_eval=_optional_str(data, "nuisance_severity_eval"),
        changing_pair_policy=(
            _optional_str(data, "changing_pair_policy") or "fallback"
        ),
    )


def _parse_embedding_models(
    data: dict[str, Any] | None,
    *,
    has_fixture: bool,
) -> tuple[EmbeddingModelConfig, ...]:
    if data is None:
        if has_fixture:
            return (
                EmbeddingModelConfig(
                    model_id="fixture_color_texture_v1",
                    kind="color_texture",
                    model_name=None,
                    batch_size=64,
                    device="cpu",
                ),
            )
        return ()
    models = _mapping_list(data, "models")
    parsed: list[EmbeddingModelConfig] = []
    for item in models:
        parsed.append(
            EmbeddingModelConfig(
                model_id=_str(item, "model_id"),
                kind=_str(item, "kind"),
                model_name=_optional_str(item, "model_name"),
                batch_size=_int(item, "batch_size"),
                device=_optional_str(item, "device") or "cpu",
            )
        )
    return tuple(parsed)


def _parse_training_config(data: dict[str, Any] | None) -> TrainingConfig:
    if data is None:
        return TrainingConfig(
            model_id="jepa_predictor_v1",
            embedding_model_id=None,
            epochs=200,
            batch_size=32,
            learning_rate=0.05,
            weight_decay=0.001,
        )
    hidden = data.get("hidden_dim")
    hidden_dim = 512 if hidden is None else int(hidden)
    return TrainingConfig(
        model_id=_str(data, "model_id"),
        embedding_model_id=_optional_str(data, "embedding_model_id"),
        epochs=_int(data, "epochs"),
        batch_size=_int(data, "batch_size"),
        learning_rate=_float(data, "learning_rate"),
        weight_decay=_float(data, "weight_decay"),
        predictor=_optional_str(data, "predictor") or "linear",
        hidden_dim=hidden_dim,
        optimizer=_optional_str(data, "optimizer") or "sgd",
    )


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


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = data.get(key)
    if value is None:
        return None
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


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        msg = f"Expected integer or null for {key}"
        raise ValueError(msg)
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"Expected non-empty string or null for {key}"
        raise ValueError(msg)
    return value


def _float(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float):
        msg = f"Expected number for {key}"
        raise ValueError(msg)
    return float(value)


def _str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        msg = f"Expected non-empty list for {key}"
        raise ValueError(msg)
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            msg = f"Expected non-empty string values for {key}"
            raise ValueError(msg)
        result.append(item)
    return result


def _mapping_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        msg = f"Expected non-empty list for {key}"
        raise ValueError(msg)
    result: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            msg = f"Expected mapping values for {key}"
            raise ValueError(msg)
        result.append(item)
    return result


def _normalize_extension(value: str) -> str:
    extension = value.lower()
    if extension.startswith("."):
        extension = extension[1:]
    if not extension:
        msg = "image extension cannot be empty"
        raise ValueError(msg)
    return extension
