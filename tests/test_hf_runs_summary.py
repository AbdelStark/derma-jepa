from __future__ import annotations

import json
from pathlib import Path

import pytest

from derma_jepa.hf_runs import summarize_run_dir, summary_as_json


def _write_run(run_dir: Path, *, with_report: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_id": run_dir.name,
        "tier": "fixture",
        "model_id": "jepa_predictor_fixture_v1",
        "primary_metric": "auroc",
        "primary_score": 0.973,
        "result": {
            "strongest_baseline": {"name": "pixel_l2", "auroc": 0.951},
            "jepa_style_model": {"metrics": {"auroc": 0.973}},
            "delta_auroc_vs_strongest_baseline": 0.022,
            "interpretation": (
                "JEPA-style predictor matched or slightly exceeded the "
                "strongest baseline."
            ),
        },
        "representation_health": {"collapsed": False},
        "training": {
            "runtime_seconds": 12.5,
            "stable_pairs_used_for_training": 16,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if with_report:
        report = {
            "acceptance_gate": {
                "passed": True,
                "criterion": "strongest fixture baseline AUROC >= 0.950",
            }
        }
        (run_dir / "benchmark_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )


def test_summarize_run_dir_parses_metrics_and_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "hf-demo"
    _write_run(run_dir)

    summary = summarize_run_dir(run_dir)

    assert summary.run_id == "hf-demo"
    assert summary.tier == "fixture"
    assert summary.model_id == "jepa_predictor_fixture_v1"
    assert summary.primary_metric == "auroc"
    assert summary.primary_score == pytest.approx(0.973)
    assert summary.strongest_baseline_name == "pixel_l2"
    assert summary.strongest_baseline_score == pytest.approx(0.951)
    assert summary.delta_vs_baseline == pytest.approx(0.022)
    assert summary.collapsed is False
    assert summary.runtime_seconds == pytest.approx(12.5)
    assert summary.train_stable_pairs == 16
    assert summary.acceptance_gate_passed is True
    assert summary.acceptance_gate_criterion
    assert summary.local_run_dir == run_dir


def test_summarize_run_dir_without_benchmark_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "hf-demo-no-report"
    _write_run(run_dir, with_report=False)

    summary = summarize_run_dir(run_dir)

    assert summary.acceptance_gate_passed is None
    assert summary.acceptance_gate_criterion is None


def test_summarize_run_dir_requires_metrics_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        summarize_run_dir(run_dir)


def test_summary_json_round_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "hf-json"
    _write_run(run_dir)

    summary = summarize_run_dir(run_dir)
    payload = json.loads(summary_as_json(summary))

    assert payload["run_id"] == "hf-json"
    assert payload["strongest_baseline"]["name"] == "pixel_l2"
    assert payload["acceptance_gate"]["passed"] is True
