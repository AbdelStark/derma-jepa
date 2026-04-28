"""Render Figure 1: cross-backbone test-AUROC gradient on `strong_held_out_2`.

Reads paper/figures/locked-numbers.json (the single source of truth) and
produces paper/figures/fig1-cross-backbone-gradient.pdf. Uses matplotlib;
no other dependencies.

Run from the repo root:
    uv run --with matplotlib python paper/figures/build_fig1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    fig_dir = Path(__file__).resolve().parent
    locked = json.loads((fig_dir / "locked-numbers.json").read_text())
    runs = locked["runs"]
    baseline = locked["baseline_strongest"]

    # Order: web → general-medical → dermoscopy, with DINOv2 first as the
    # canonical natural-image baseline before the CLIP-architecture rows.
    rows = [
        ("DINOv2 ViT-B/14\n(LVD-142M, web)", runs["exp004_dinov2_linear"], False),
        ("OpenAI CLIP ViT-B/16\n(WIT, web)", runs["exp006b_clip_linear"], False),
        (
            "BiomedCLIP ViT-B/16\n(PMC-15M, gen. medical)",
            runs["exp008_biomedclip_linear"],
            True,
        ),
        (
            "DermLIP ViT-B/16\n(Derm1M, dermatology)",
            runs["exp007_dermlip_linear"],
            True,
        ),
    ]

    labels = [name for name, _, _ in rows]
    means = []
    err_low = []
    err_high = []
    is_seed_sweep = []
    for _, entry, has_seeds in rows:
        if has_seeds:
            means.append(entry["test_auroc_seed_mean"])
            err_low.append(entry["test_auroc_seed_mean"] - entry["test_auroc_min"])
            err_high.append(entry["test_auroc_max"] - entry["test_auroc_seed_mean"])
        else:
            means.append(entry["test_auroc"])
            err_low.append(entry["test_auroc"] - entry["test_ci_low"])
            err_high.append(entry["test_ci_high"] - entry["test_auroc"])
        is_seed_sweep.append(has_seeds)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(rows))
    colors = ["#9aa3b2", "#9aa3b2", "#6c95cc", "#3d8c5c"]
    ax.barh(
        y,
        means,
        xerr=[err_low, err_high],
        color=colors,
        edgecolor="#222",
        linewidth=0.6,
        error_kw={"ecolor": "#222", "capsize": 3, "lw": 0.8},
    )
    ax.set_yticks(y, labels=labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Test AUROC on `strong_held_out_2` (n=2,000 pairs)")

    # Pixel-L2 reference line.
    ax.axvline(
        baseline["auroc"],
        linestyle="--",
        linewidth=0.9,
        color="#888",
    )
    ax.text(
        baseline["auroc"] + 0.005,
        -0.45,
        f"pixel L2 = {baseline['auroc']:.3f}",
        fontsize=8,
        color="#666",
        va="top",
    )

    # Random-baseline reference at 0.5.
    ax.axvline(0.5, linestyle=":", linewidth=0.7, color="#aaa")
    ax.text(0.5 + 0.005, -0.4, "random = 0.5", fontsize=8, color="#888", va="top")

    # Number labels at end of each bar.
    for i, (mean, has_seeds) in enumerate(zip(means, is_seed_sweep, strict=True)):
        if has_seeds:
            std = rows[i][1]["test_auroc_seed_std"]
            label = f"  {mean:.3f} ± {std:.3f}  (5 seeds)"
        else:
            label = f"  {mean:.3f}  (1 seed)"
        ax.text(mean, i, label, va="center", fontsize=8.5)

    # Contamination caveat annotation on the DermLIP bar.
    dermlip_idx = len(rows) - 1
    ax.annotate(
        "[contamination caveat:\nDerm1M ⊇ HAM10000]",
        xy=(rows[dermlip_idx][1]["test_auroc_seed_mean"], dermlip_idx),
        xytext=(0.40, dermlip_idx - 0.55),
        fontsize=7.5,
        color="#922",
        arrowprops={"arrowstyle": "-", "color": "#922", "lw": 0.5},
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()

    out = fig_dir / "fig1-cross-backbone-gradient.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
