"""Render Figure 2: train-vs-test AUROC scatter on the EXP-004 proxy.

Reads each primary-tier run's metrics.json from outputs/hf-runs/<run-id>/
and produces paper/figures/fig2-train-vs-test-scatter.pdf. The plot has a
y = x reference line, a random-baseline reference at y = 0.5, and one point
per run labelled with the experiment ID. Runs that fit training to ceiling
but test below random sit on the vertical near x = 1, y < 0.5; runs whose
test tracks training cluster on the diagonal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "outputs" / "hf-runs"

RUNS = [
    ("EXP-004",  "ham10000-hf-dinov2-exp004-v1",     "DINOv2 / linear / SGD",   "#9aa3b2"),
    ("EXP-005",  "ham10000-hf-dinov2-exp005-v1",     "DINOv2 / MLP / SGD",      "#9aa3b2"),
    ("EXP-006a", "ham10000-hf-dinov2-exp006a-v1",    "DINOv2 / MLP / Adam",     "#9aa3b2"),
    ("EXP-006b", "ham10000-hf-clip-exp006b-v1",      "CLIP / linear / SGD",     "#9aa3b2"),
    ("EXP-008",  "ham10000-hf-biomedclip-exp008-v1", "BiomedCLIP / linear",     "#6c95cc"),
    ("EXP-007",  "ham10000-hf-dermlip-exp007-v1",    "DermLIP / linear",        "#3d8c5c"),
]


def main() -> int:
    fig_dir = Path(__file__).resolve().parent

    points = []
    for label, run_id, descr, colour in RUNS:
        m = json.loads((RUNS_ROOT / run_id / "metrics.json").read_text())
        train = m["splits"]["train"]["jepa_style_model"]["auroc"]
        test = m["splits"]["test"]["jepa_style_model"]["auroc"]
        points.append((label, descr, train, test, colour))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    # y = x reference.
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="#bbb",
            label="train $=$ test")
    # Random baseline at y = 0.5.
    ax.axhline(0.5, linestyle=":", linewidth=0.7, color="#aaa")
    ax.text(0.02, 0.51, "random = 0.5", fontsize=7.5, color="#888")

    # Per-point label placement (text x, text y); connector arrows used
    # for the bottom-right cluster (EXP-004 / EXP-006a / EXP-006b) to
    # avoid mutual occlusion.
    placements = {
        "EXP-005":  ((0.30,  0.18), False),
        "EXP-004":  ((0.55,  0.05), True),
        "EXP-006a": ((0.55,  0.13), True),
        "EXP-006b": ((0.85,  0.08), True),
        "EXP-008":  ((0.62,  0.40), True),
        "EXP-007":  ((0.50,  0.85), True),
    }

    for label, descr, train, test, colour in points:
        ax.scatter(train, test, s=70, color=colour, edgecolors="#222",
                   linewidth=0.6, zorder=3)
        (tx, ty), use_arrow = placements[label]
        if use_arrow:
            ax.annotate(
                f"{label}: {descr}",
                xy=(train, test),
                xytext=(tx, ty),
                fontsize=7.5,
                color="#222",
                arrowprops={"arrowstyle": "-", "color": "#888", "lw": 0.4,
                            "shrinkA": 1, "shrinkB": 4},
                ha="left",
            )
        else:
            ax.text(tx, ty, f"{label}: {descr}", fontsize=7.5, color="#222")

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Train AUROC (mixed-family training)")
    ax.set_ylabel("Test AUROC on `strong_held_out_2`")
    ax.set_title("Train-vs-test AUROC on the EXP-004 proxy",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = fig_dir / "fig2-train-vs-test-scatter.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
