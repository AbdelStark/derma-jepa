from __future__ import annotations

from pathlib import Path
from typing import Any

from derma_jepa.run import read_json, require_run_file, write_json

REQUIRED_RUN_FILES = (
    "config.yaml",
    "manifest_train.parquet",
    "manifest_val.parquet",
    "manifest_test.parquet",
    "metrics.json",
    "baseline_metrics.json",
    "model_card.md",
    "environment.txt",
    "artifacts/embeddings/fixture_embeddings.npz",
    "artifacts/embeddings/fixture_embeddings.parquet",
    "artifacts/embeddings/embedding_index.json",
    "artifacts/plots/baseline_score_histogram.png",
    "artifacts/reports/baseline_failure_cases.json",
    "logs/train.log",
    "logs/eval.log",
)


def validate_fixture_run(run_dir: Path, *, allow_negative_result: bool = False) -> Path:
    for relative in REQUIRED_RUN_FILES:
        require_run_file(run_dir, relative)
    baseline_metrics = read_json(run_dir / "baseline_metrics.json")
    metrics = read_json(run_dir / "metrics.json")
    strongest = baseline_metrics["strongest_baseline"]
    auroc = float(strongest["auroc"])
    passed = auroc >= 0.95 or allow_negative_result
    if not passed:
        msg = (
            f"Fixture acceptance gate failed: strongest baseline AUROC {auroc:.3f} "
            "is below 0.950"
        )
        raise RuntimeError(msg)
    report: dict[str, Any] = {
        "run_id": metrics["run_id"],
        "tier": "fixture",
        "required_files_checked": list(REQUIRED_RUN_FILES),
        "strongest_baseline": strongest,
        "acceptance_gate": {
            "passed": passed,
            "criterion": "strongest fixture baseline AUROC >= 0.950",
            "allow_negative_result": allow_negative_result,
        },
    }
    out_path = run_dir / "benchmark_report.json"
    write_json(out_path, report)
    return out_path
