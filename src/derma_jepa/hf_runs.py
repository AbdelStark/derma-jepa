"""Fetch and summarize DermaJEPA run artifacts uploaded to the Hugging Face Hub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from derma_jepa.run import read_json

_REPO_TYPES = frozenset({"dataset", "model", "space"})


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    tier: str | None
    model_id: str | None
    primary_metric: str | None
    primary_score: float | None
    strongest_baseline_name: str | None
    strongest_baseline_score: float | None
    delta_vs_baseline: float | None
    interpretation: str | None
    collapsed: bool | None
    runtime_seconds: float | None
    train_stable_pairs: int | None
    acceptance_gate_passed: bool | None
    acceptance_gate_criterion: str | None
    local_run_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "model_id": self.model_id,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "strongest_baseline": {
                "name": self.strongest_baseline_name,
                "score": self.strongest_baseline_score,
            },
            "delta_vs_baseline": self.delta_vs_baseline,
            "interpretation": self.interpretation,
            "collapsed": self.collapsed,
            "runtime_seconds": self.runtime_seconds,
            "train_stable_pairs": self.train_stable_pairs,
            "acceptance_gate": {
                "passed": self.acceptance_gate_passed,
                "criterion": self.acceptance_gate_criterion,
            },
            "local_run_dir": str(self.local_run_dir),
        }

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append(f"run_id: {self.run_id}")
        if self.tier:
            lines.append(f"tier: {self.tier}")
        if self.model_id:
            lines.append(f"model_id: {self.model_id}")
        if self.primary_metric and self.primary_score is not None:
            lines.append(f"{self.primary_metric}: {self.primary_score:.4f}")
        if (
            self.strongest_baseline_name is not None
            and self.strongest_baseline_score is not None
        ):
            lines.append(
                "strongest_baseline: "
                f"{self.strongest_baseline_name} = "
                f"{self.strongest_baseline_score:.4f}"
            )
        if self.delta_vs_baseline is not None:
            lines.append(f"delta_vs_baseline: {self.delta_vs_baseline:+.4f}")
        if self.interpretation:
            lines.append(f"interpretation: {self.interpretation}")
        if self.collapsed is not None:
            lines.append(f"collapsed: {self.collapsed}")
        if self.runtime_seconds is not None:
            lines.append(f"runtime_seconds: {self.runtime_seconds:.2f}")
        if self.train_stable_pairs is not None:
            lines.append(f"train_stable_pairs: {self.train_stable_pairs}")
        if self.acceptance_gate_passed is not None:
            gate = "passed" if self.acceptance_gate_passed else "failed"
            lines.append(f"acceptance_gate: {gate}")
            if self.acceptance_gate_criterion:
                lines.append(f"acceptance_criterion: {self.acceptance_gate_criterion}")
        lines.append(f"local_run_dir: {self.local_run_dir}")
        return "\n".join(lines)


def fetch_run_dir(
    *,
    repo_id: str,
    run_id: str,
    dest: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
    path_in_repo: str | None = None,
) -> Path:
    """Download the subfolder for a single run from the Hugging Face Hub."""
    if repo_type not in _REPO_TYPES:
        msg = f"repo_type must be one of {sorted(_REPO_TYPES)}"
        raise ValueError(msg)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = (
            "huggingface-hub is required for `derma-jepa hf-run` commands. "
            "Install it with `uv pip install 'huggingface-hub>=1.0'`."
        )
        raise RuntimeError(msg) from exc

    subdir = path_in_repo if path_in_repo is not None else run_id
    allow_pattern = f"{subdir}/**"
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=[allow_pattern],
    )
    local_run_dir = dest / subdir
    if not local_run_dir.exists():
        msg = (
            f"Expected run directory not found after download: {local_run_dir}. "
            "Verify repo_id, run_id, and path_in_repo."
        )
        raise FileNotFoundError(msg)
    return local_run_dir


def summarize_run_dir(run_dir: Path) -> RunSummary:
    """Build a compact summary from a downloaded run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        msg = f"metrics.json not found in run directory: {run_dir}"
        raise FileNotFoundError(msg)
    metrics = read_json(metrics_path)
    result = metrics.get("result") or {}
    strongest = result.get("strongest_baseline") or {}
    health = metrics.get("representation_health") or {}
    training = metrics.get("training") or {}

    acceptance_passed: bool | None = None
    acceptance_criterion: str | None = None
    report_path = run_dir / "benchmark_report.json"
    if report_path.exists():
        report = read_json(report_path)
        gate = report.get("acceptance_gate") or {}
        if isinstance(gate.get("passed"), bool):
            acceptance_passed = bool(gate["passed"])
        criterion = gate.get("criterion")
        if isinstance(criterion, str):
            acceptance_criterion = criterion

    return RunSummary(
        run_id=str(metrics.get("run_id", run_dir.name)),
        tier=_optional_str(metrics.get("tier")),
        model_id=_optional_str(metrics.get("model_id")),
        primary_metric=_optional_str(metrics.get("primary_metric")),
        primary_score=_optional_float(metrics.get("primary_score")),
        strongest_baseline_name=_optional_str(strongest.get("name")),
        strongest_baseline_score=_optional_float(strongest.get("auroc")),
        delta_vs_baseline=_optional_float(
            result.get("delta_auroc_vs_strongest_baseline")
        ),
        interpretation=_optional_str(result.get("interpretation")),
        collapsed=bool(health["collapsed"]) if "collapsed" in health else None,
        runtime_seconds=_optional_float(training.get("runtime_seconds")),
        train_stable_pairs=_optional_int(
            training.get("stable_pairs_used_for_training")
        ),
        acceptance_gate_passed=acceptance_passed,
        acceptance_gate_criterion=acceptance_criterion,
        local_run_dir=run_dir,
    )


def fetch_and_summarize(
    *,
    repo_id: str,
    run_id: str,
    dest: Path,
    repo_type: str = "dataset",
    revision: str | None = None,
    path_in_repo: str | None = None,
) -> RunSummary:
    local_run_dir = fetch_run_dir(
        repo_id=repo_id,
        run_id=run_id,
        dest=dest,
        repo_type=repo_type,
        revision=revision,
        path_in_repo=path_in_repo,
    )
    return summarize_run_dir(local_run_dir)


def summary_as_json(summary: RunSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2, sort_keys=True)


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
