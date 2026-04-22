from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from derma_jepa.contracts import read_manifest
from derma_jepa.run import read_json, require_run_file, write_json


def export_demo_bundle(run_dir: Path, out_dir: Path) -> Path:
    require_run_file(run_dir, "baseline_metrics.json")
    require_run_file(run_dir, "metrics.json")
    require_run_file(run_dir, "manifest_test.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_json(run_dir / "metrics.json")
    baselines = read_json(run_dir / "baseline_metrics.json")
    rows = read_manifest(run_dir / "manifest_test.parquet")
    selected = rows[: min(4, len(rows))]
    case_images = _copy_case_images(selected, image_dir)
    payload: dict[str, Any] = {
        "run_id": metrics["run_id"],
        "demo_type": "deterministic_fixture_research_demo",
        "safety_boundary": (
            "research demo; longitudinal-proxy task; not diagnostic; not medical advice"
        ),
        "provenance": {
            "run_dir": str(run_dir),
            "source_dataset": selected[0].source_dataset if selected else "unknown",
            "split": "test",
            "model_id": metrics["model_id"],
            "strongest_baseline": baselines["strongest_baseline"],
        },
        "timeline": [
            {
                "pair_id": row.pair_id,
                "label_hidden_by_default": row.pair_label,
                "context_image_id": row.context_image_id,
                "target_image_id": row.target_image_id,
                "context_image": case_images[row.context_image_id],
                "target_image": case_images[row.target_image_id],
                "source_metadata": {
                    "source_dataset": row.source_dataset,
                    "anatomical_site": row.anatomical_site,
                    "diagnosis": row.diagnosis,
                    "preprocessing_profile": row.preprocessing_profile,
                },
            }
            for row in selected
        ],
        "drift_chart": _drift_chart(baselines),
        "nuisance_stress": [
            {
                "family": "brightness_contrast_blur",
                "severity": "mild",
                "summary": (
                    "Stable fixture pairs are generated after split with "
                    "recorded nuisance parameters."
                ),
            }
        ],
        "failure_cases": _failure_cases(baselines),
    }
    write_json(out_dir / "demo_case.json", payload)
    write_json(run_dir / "artifacts" / "demo_cases" / "fixture_case.json", payload)
    _write_html(out_dir / "index.html", payload)
    return out_dir / "demo_case.json"


def validate_demo_artifact(artifact_dir: Path) -> Path:
    case_path = artifact_dir / "demo_case.json"
    index_path = artifact_dir / "index.html"
    if not case_path.exists() or not index_path.exists():
        msg = f"Demo artifact is incomplete: {artifact_dir}"
        raise FileNotFoundError(msg)
    return index_path


def _copy_case_images(rows: list[Any], image_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for row in rows:
        for image_id, source in (
            (row.context_image_id, Path(row.context_path)),
            (row.target_image_id, Path(row.target_path)),
        ):
            if image_id in copied:
                continue
            target = image_dir / f"{image_id}.png"
            shutil.copyfile(source, target)
            copied[image_id] = str(target.relative_to(image_dir.parent))
    return copied


def _drift_chart(baselines: dict[str, Any]) -> list[dict[str, Any]]:
    chart: list[dict[str, Any]] = []
    for baseline_name, payload in baselines["baselines"].items():
        for item in payload["pair_scores"]:
            chart.append(
                {
                    "baseline": baseline_name,
                    "pair_id": item["pair_id"],
                    "label": item["label"],
                    "score": item["score"],
                }
            )
    return chart


def _failure_cases(baselines: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for baseline_name, payload in baselines["baselines"].items():
        sorted_scores = sorted(
            payload["pair_scores"],
            key=lambda item: float(item["score"]),
        )
        stable_high = [
            item for item in reversed(sorted_scores) if item["label"] == "stable"
        ]
        changing_low = [item for item in sorted_scores if item["label"] == "changing"]
        if stable_high:
            cases.append(
                {
                    "baseline": baseline_name,
                    "failure_mode": "stable_high_score",
                    **stable_high[0],
                }
            )
        if changing_low:
            cases.append(
                {
                    "baseline": baseline_name,
                    "failure_mode": "changing_low_score",
                    **changing_low[0],
                }
            )
    return cases


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    strongest = payload["provenance"]["strongest_baseline"]
    rows = "\n".join(
        f"<tr><td>{item['baseline']}</td><td>{item['pair_id']}</td><td>{item['label']}</td>"
        f"<td>{float(item['score']):.4f}</td></tr>"
        for item in payload["drift_chart"]
    )
    timeline = "\n".join(
        "<figure>"
        f'<img src="{case["context_image"]}" '
        f'alt="{case["context_image_id"]}">'
        f'<img src="{case["target_image"]}" alt="{case["target_image_id"]}">'
        f"<figcaption>{case['pair_id']}</figcaption></figure>"
        for case in payload["timeline"]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DermaJEPA Fixture Demo</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 32px;
      color: #1f2933;
    }}
    header {{ max-width: 920px; }}
    .boundary {{
      padding: 12px 16px;
      background: #eef2ff;
      border-left: 4px solid #4056a1;
    }}
    figure {{ display: inline-block; margin: 12px 16px 12px 0; }}
    img {{
      width: 112px;
      height: 112px;
      image-rendering: auto;
      border: 1px solid #ccd2dc;
    }}
    table {{ border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #d7dde8; padding: 8px 10px; text-align: left; }}
  </style>
</head>
<body>
  <header>
    <h1>DermaJEPA Fixture Research Demo</h1>
    <p class="boundary">{payload["safety_boundary"]}</p>
    <p>Run <strong>{payload["run_id"]}</strong>. Strongest fixture baseline:
    <strong>{strongest["name"]}</strong> AUROC {float(strongest["auroc"]):.3f}.</p>
  </header>
  <section>
    <h2>Case Timeline</h2>
    {timeline}
  </section>
  <section>
    <h2>Drift Scores</h2>
    <table>
      <thead>
        <tr><th>Baseline</th><th>Pair</th><th>Proxy Label</th><th>Score</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
