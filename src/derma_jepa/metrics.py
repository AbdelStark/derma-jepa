from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class MetricSummary:
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    auprc: float
    eer_threshold: float
    equal_error_rate: float
    fpr_at_fixed_tpr: float
    fixed_tpr: float
    bootstrap_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def binary_metric_summary(
    labels: list[int],
    scores: list[float],
    *,
    bootstrap_samples: int,
    ci_level: float,
    fixed_tpr: float,
    seed: int,
) -> MetricSummary:
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    _validate_binary_inputs(y_true, y_score)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    ci_low, ci_high = bootstrap_auroc_ci(
        y_true,
        y_score,
        samples=bootstrap_samples,
        ci_level=ci_level,
        seed=seed,
    )
    threshold, eer = equal_error_threshold(y_true, y_score)
    fpr_at_tpr = fpr_at_fixed_tpr(y_true, y_score, fixed_tpr)
    return MetricSummary(
        auroc=auroc,
        auroc_ci_low=ci_low,
        auroc_ci_high=ci_high,
        auprc=auprc,
        eer_threshold=threshold,
        equal_error_rate=eer,
        fpr_at_fixed_tpr=fpr_at_tpr,
        fixed_tpr=fixed_tpr,
        bootstrap_samples=bootstrap_samples,
    )


def bootstrap_auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    indices = np.arange(labels.shape[0])
    for _ in range(samples):
        sample = rng.choice(indices, size=indices.shape[0], replace=True)
        y_sample = labels[sample]
        if np.unique(y_sample).shape[0] < 2:
            continue
        values.append(float(roc_auc_score(y_sample, scores[sample])))
    if not values:
        point = float(roc_auc_score(labels, scores))
        return point, point
    alpha = 1.0 - ci_level
    low = float(np.quantile(values, alpha / 2))
    high = float(np.quantile(values, 1 - alpha / 2))
    return low, high


def equal_error_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2)
    return float(thresholds[index]), eer


def fpr_at_fixed_tpr(labels: np.ndarray, scores: np.ndarray, fixed_tpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.flatnonzero(tpr >= fixed_tpr)
    if valid.size == 0:
        return 1.0
    return float(np.min(fpr[valid]))


def _validate_binary_inputs(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.ndim != 1 or scores.ndim != 1:
        msg = "labels and scores must be one-dimensional"
        raise ValueError(msg)
    if labels.shape[0] != scores.shape[0]:
        msg = "labels and scores must have the same length"
        raise ValueError(msg)
    if labels.shape[0] < 2:
        msg = "at least two examples are required"
        raise ValueError(msg)
    if set(np.unique(labels).tolist()) != {0, 1}:
        msg = "labels must contain both stable=0 and changing=1 examples"
        raise ValueError(msg)
