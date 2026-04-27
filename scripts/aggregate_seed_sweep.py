"""Aggregate test/val/train AUROC across seed-sweep runs.

Pulls the run summaries from a list of run_ids, prints per-seed numbers, and
reports mean / std / min / max + parametric and bootstrap CIs across seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_metrics(run_dir: Path) -> dict[str, dict[str, float]] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return None
    data = json.loads(metrics_path.read_text())
    splits = data.get("splits", {})
    out: dict[str, dict[str, float]] = {}
    for split_name, payload in splits.items():
        jepa = payload.get("jepa_style_model", {})
        if "auroc" not in jepa:
            continue
        out[split_name] = {
            "auroc": float(jepa["auroc"]),
            "auroc_ci_low": float(jepa["auroc_ci_low"]),
            "auroc_ci_high": float(jepa["auroc_ci_high"]),
        }
    return out


def summarise(label: str, values: list[float]) -> str:
    if not values:
        return f"{label}: (no data)"
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    sem = std / math.sqrt(len(values)) if len(values) >= 2 else 0.0
    half_width = 1.96 * sem
    return (
        f"{label}: n={len(values)}  mean={mean:.4f}  std={std:.4f}  "
        f"min={min(values):.4f}  max={max(values):.4f}  "
        f"95%CI[mean]=[{mean - half_width:.4f}, {mean + half_width:.4f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=Path("outputs/hf-runs"),
        help="local mirror directory for the HF runs dataset",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="run id to include (repeat for each seed run)",
    )
    parser.add_argument(
        "--label",
        default="sweep",
        help="label to prefix in the report",
    )
    args = parser.parse_args()

    rows: list[tuple[str, dict[str, dict[str, float]]]] = []
    missing: list[str] = []
    for run_id in args.run_id:
        run_dir = args.mirror_root / run_id
        metrics = load_metrics(run_dir)
        if metrics is None:
            missing.append(run_id)
            continue
        rows.append((run_id, metrics))

    if missing:
        print(
            "WARNING: missing local mirror for runs: "
            + ", ".join(missing)
            + "\n  Pull them with `derma-jepa hf-run summary --run-id <id>` first.",
            file=sys.stderr,
        )

    if not rows:
        print("No runs available; aborting.", file=sys.stderr)
        return 1

    print(f"=== {args.label} — per-seed AUROCs ===")
    header = f"{'run_id':60s} {'train':>10s} {'val':>10s} {'test':>10s}"
    print(header)
    print("-" * len(header))
    train_vals: list[float] = []
    val_vals: list[float] = []
    test_vals: list[float] = []
    for run_id, metrics in rows:
        t = metrics.get("train", {}).get("auroc")
        v = metrics.get("val", {}).get("auroc")
        s = metrics.get("test", {}).get("auroc")
        if t is not None:
            train_vals.append(t)
        if v is not None:
            val_vals.append(v)
        if s is not None:
            test_vals.append(s)
        print(
            f"{run_id:60s} "
            f"{t if t is not None else float('nan'):>10.4f} "
            f"{v if v is not None else float('nan'):>10.4f} "
            f"{s if s is not None else float('nan'):>10.4f}"
        )

    print()
    print(f"=== {args.label} — across-seed summary ===")
    print(summarise("train AUROC", train_vals))
    print(summarise("val   AUROC", val_vals))
    print(summarise("test  AUROC", test_vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
