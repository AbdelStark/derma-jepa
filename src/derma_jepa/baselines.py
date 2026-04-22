from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from derma_jepa.config import PipelineConfig
from derma_jepa.contracts import ManifestRow, read_manifest
from derma_jepa.metrics import binary_metric_summary
from derma_jepa.preprocessing import load_preprocessed_rgb
from derma_jepa.run import append_log, prepare_run_dir, write_json


def evaluate_baselines(config: PipelineConfig, split: str = "test") -> Path:
    run_dir = prepare_run_dir(config)
    rows = list(read_manifest(run_dir / f"manifest_{split}.parquet"))
    labels = [row.label_int for row in rows]
    pixel_scores = [_pixel_l2(row, config.preprocessing.image_size) for row in rows]
    ssim_scores = [_ssim_distance(row, config.preprocessing.image_size) for row in rows]

    baselines = {
        "pixel_l2": _baseline_payload(
            config, labels, pixel_scores, rows, seed_offset=11
        ),
        "ssim_distance": _baseline_payload(
            config, labels, ssim_scores, rows, seed_offset=23
        ),
    }
    strongest = max(baselines.items(), key=lambda item: item[1]["metrics"]["auroc"])
    payload: dict[str, Any] = {
        "run_id": config.run_id,
        "tier": config.tier,
        "split": split,
        "task": "longitudinal_proxy_change_detection",
        "positive_label": "changing",
        "baselines": baselines,
        "strongest_baseline": {
            "name": strongest[0],
            "auroc": strongest[1]["metrics"]["auroc"],
            "auroc_ci_low": strongest[1]["metrics"]["auroc_ci_low"],
            "auroc_ci_high": strongest[1]["metrics"]["auroc_ci_high"],
        },
        "clinical_boundary": (
            "research demo only; not diagnostic and not medical advice"
        ),
    }
    write_json(run_dir / "baseline_metrics.json", payload)
    write_json(run_dir / "metrics.json", _metrics_payload(config, payload))
    _write_model_card(run_dir, payload)
    _write_train_log(run_dir, config.tier)
    _write_score_plot(
        run_dir / "artifacts" / "plots" / "baseline_score_histogram.png",
        baselines,
    )
    append_log(
        run_dir, "eval.log", f"evaluated {config.tier} baselines on {split} split"
    )
    return run_dir / "baseline_metrics.json"


def evaluate_fixture_baselines(config: PipelineConfig, split: str = "test") -> Path:
    return evaluate_baselines(config, split=split)


def _baseline_payload(
    config: PipelineConfig,
    labels: list[int],
    scores: list[float],
    rows: list[ManifestRow],
    *,
    seed_offset: int,
) -> dict[str, Any]:
    metrics = binary_metric_summary(
        labels,
        scores,
        bootstrap_samples=config.metrics.bootstrap_samples,
        ci_level=config.metrics.ci_level,
        fixed_tpr=config.metrics.fixed_tpr,
        seed=config.seed + seed_offset,
    )
    return {
        "metrics": metrics.to_dict(),
        "pair_scores": [
            {
                "pair_id": row.pair_id,
                "label": row.pair_label,
                "score": score,
                "context_image_id": row.context_image_id,
                "target_image_id": row.target_image_id,
            }
            for row, score in zip(rows, scores, strict=True)
        ],
    }


def _metrics_payload(
    config: PipelineConfig, baseline_payload: dict[str, Any]
) -> dict[str, Any]:
    strongest = baseline_payload["strongest_baseline"]
    return {
        "run_id": config.run_id,
        "tier": config.tier,
        "model_id": f"baseline_only_{config.tier}_contract",
        "primary_metric": "auroc",
        "primary_score": "latent_drift_proxy_not_yet_trained",
        "status": f"baseline_only_{config.tier}_milestone",
        "result": {
            "strongest_baseline": strongest,
            "jepa_style_model": None,
            "interpretation": (
                "This run validates the manifest, preprocessing, metric, and "
                "cheap-baseline path. No JEPA-style model result is claimed."
            ),
        },
        "clinical_boundary": (
            "research demo only; not diagnostic and not validated for patient use"
        ),
    }


def _pixel_l2(row: ManifestRow, image_size: int) -> float:
    left = load_preprocessed_rgb(Path(row.context_path), image_size)
    right = load_preprocessed_rgb(Path(row.target_path), image_size)
    return float(np.sqrt(np.mean(np.square(left - right))))


def _ssim_distance(row: ManifestRow, image_size: int) -> float:
    left = load_preprocessed_rgb(Path(row.context_path), image_size).mean(axis=2)
    right = load_preprocessed_rgb(Path(row.target_path), image_size).mean(axis=2)
    similarity = structural_similarity(  # type: ignore[no-untyped-call]
        left,
        right,
        data_range=1.0,
    )
    return float(1.0 - similarity)


def _write_model_card(run_dir: Path, payload: dict[str, Any]) -> None:
    strongest = payload["strongest_baseline"]
    tier = str(payload["tier"])
    limitation = (
        "The fixture dataset is synthetic and exists to prove schemas, "
        "preprocessing, metrics, run artifacts, and demo export."
        if tier == "fixture"
        else (
            "The public-data proxy uses local research dataset files and cheap "
            "baselines only; it establishes the audit and evaluation path before "
            "model claims."
        )
    )
    text = f"""# {str(payload["tier"]).title()} baseline model card

## Scope

This run validates the DermaJEPA {tier}-tier artifact contract. It
evaluates pixel-space baselines on a longitudinal-proxy task.

## Evidence

- Run ID: `{payload["run_id"]}`
- Split: `{payload["split"]}`
- Strongest baseline: `{strongest["name"]}`
- AUROC: `{strongest["auroc"]:.3f}`
- Bootstrap CI: `[{strongest["auroc_ci_low"]:.3f}, {strongest["auroc_ci_high"]:.3f}]`

## Limitations

No JEPA-style model is trained in this run. {limitation} It is not diagnostic,
not medical advice, and not validated for patient use.
"""
    (run_dir / "model_card.md").write_text(text, encoding="utf-8")


def _write_train_log(run_dir: Path, tier: str) -> None:
    append_log(
        run_dir,
        "train.log",
        (
            f"{tier} tier has no model training; JEPA-style predictor training "
            "begins in Milestone 3"
        ),
    )


def _write_score_plot(path: Path, baselines: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 720, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    margin = 54
    plot_w = width - margin * 2
    plot_h = height - margin * 2
    draw.rectangle(
        (margin, margin, margin + plot_w, margin + plot_h),
        outline=(30, 30, 30),
    )
    colors = {"stable": (48, 119, 188), "changing": (201, 76, 55)}

    for baseline_index, (name, payload) in enumerate(baselines.items()):
        y_origin = margin + 50 + baseline_index * 145
        draw.text((margin, y_origin - 34), name, fill=(20, 20, 20))
        pair_scores = payload["pair_scores"]
        max_score = max(float(item["score"]) for item in pair_scores) or 1.0
        for item_index, item in enumerate(pair_scores):
            bar_w = int((float(item["score"]) / max_score) * (plot_w - 160))
            x0 = margin + 110
            y0 = y_origin + item_index * 12
            color = colors[str(item["label"])]
            draw.rectangle((x0, y0, x0 + bar_w, y0 + 8), fill=color)
            draw.text((margin + 6, y0 - 2), str(item["label"])[:8], fill=(30, 30, 30))
    image.save(path)
