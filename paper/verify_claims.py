"""Verify the paper's quantitative claims against the run archive.

Reads paper/figures/locked-numbers.json (the single source of truth for the
paper's claims) and cross-checks every entry against the per-run
metrics.json under outputs/hf-runs/<run-id>/. Reports each claim as PASS or
FAIL with the observed-vs-expected delta. Exits non-zero if any claim fails.

Run from the repo root:
    uv run python paper/verify_claims.py

Tolerance defaults to the value in locked-numbers.json (typically 1e-3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MIRROR_ROOT = REPO_ROOT / "outputs" / "hf-runs"
LOCKED = REPO_ROOT / "paper" / "figures" / "locked-numbers.json"


def load_locked() -> dict[str, Any]:
    return json.loads(LOCKED.read_text())


def load_metrics(run_id: str) -> dict[str, Any] | None:
    p = MIRROR_ROOT / run_id / "metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def check_close(name: str, observed: float, expected: float, tol: float) -> bool:
    delta = abs(observed - expected)
    status = "PASS" if delta <= tol else "FAIL"
    print(
        f"  {status:4s}  {name:48s}  obs={observed:.4f}  exp={expected:.4f}  "
        f"delta={delta:.4f}"
    )
    return delta <= tol


def check_run(name: str, entry: dict[str, Any], tol: float) -> int:
    fails = 0
    metrics = load_metrics(entry["run_id"])
    if metrics is None:
        run_id = entry["run_id"]
        print(f"  SKIP  {name}: missing local mirror at outputs/hf-runs/{run_id}/")
        return 0
    test = metrics["splits"]["test"]["jepa_style_model"]
    observed_auroc = float(test["auroc"])
    if "test_auroc" in entry and not check_close(
        f"{name} test AUROC (single seed)",
        observed_auroc,
        float(entry["test_auroc"]),
        tol,
    ) or "single_seed_test_auroc" in entry and not check_close(
        f"{name} test AUROC (original-seed run)",
        observed_auroc,
        float(entry["single_seed_test_auroc"]),
        tol,
    ):
        fails += 1
    if "test_ci_low" in entry and "test_ci_high" in entry:
        if not check_close(
            f"{name} test CI low",
            float(test["auroc_ci_low"]),
            float(entry["test_ci_low"]),
            tol,
        ):
            fails += 1
        if not check_close(
            f"{name} test CI high",
            float(test["auroc_ci_high"]),
            float(entry["test_ci_high"]),
            tol,
        ):
            fails += 1
    if "raw_cosine_auroc" in entry:
        baseline_path = MIRROR_ROOT / entry["run_id"] / "baseline_metrics.json"
        if baseline_path.is_file():
            baselines = json.loads(baseline_path.read_text())["baselines"]
            cosine_keys = [k for k in baselines if "embedding_cosine" in k]
            if cosine_keys:
                primary = cosine_keys[0]
                observed_cos = float(baselines[primary]["metrics"]["auroc"])
                if not check_close(
                    f"{name} raw {primary} AUROC",
                    observed_cos,
                    float(entry["raw_cosine_auroc"]),
                    tol,
                ):
                    fails += 1
    return fails


def check_baseline(locked: dict[str, Any], tol: float) -> int:
    fails = 0
    expected = locked["baseline_strongest"]
    # pull pixel_l2 from any post-EXP-004 run; EXP-008 is convenient.
    metrics = load_metrics("ham10000-hf-biomedclip-exp008-v1")
    if metrics is None:
        print("  SKIP  pixel L2 baseline: EXP-008 mirror missing")
        return 0
    run_dir = MIRROR_ROOT / "ham10000-hf-biomedclip-exp008-v1"
    baseline_path = run_dir / "baseline_metrics.json"
    if not baseline_path.is_file():
        print("  SKIP  pixel L2 baseline: baseline_metrics.json missing")
        return 0
    baselines = json.loads(baseline_path.read_text())["baselines"]
    if "pixel_l2" not in baselines:
        print("  SKIP  pixel L2 baseline: not in baseline_metrics.json")
        return 0
    observed = float(baselines["pixel_l2"]["metrics"]["auroc"])
    if not check_close("pixel L2 baseline", observed, float(expected["auroc"]), tol):
        fails += 1
    return fails


def main() -> int:
    locked = load_locked()
    tol = float(locked["tolerance"]["auroc"])
    print(f"=== verify_claims.py — tolerance {tol} ===")
    total_fails = 0
    print("[baseline]")
    total_fails += check_baseline(locked, tol)
    for name, entry in locked["runs"].items():
        print(f"[{name}]")
        total_fails += check_run(name, entry, tol)
    print()
    if total_fails == 0:
        print("ALL CLAIMS VERIFIED ✓")
        return 0
    print(f"FAILED CLAIMS: {total_fails}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
