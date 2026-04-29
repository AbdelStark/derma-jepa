"""Render Figure 3: 3-panel pair-score histograms on the test split.

Reads jepa_predictor_latents.npz from the CLIP, BiomedCLIP, and DermLIP
runs. The score column is the cosine *distance* between the predicted
target and the observed target on each test pair; lower scores mean the
predictor judged the pair more similar. A correctly-oriented predictor
puts stable-pair scores below changing-pair scores; an inverted
predictor flips the ordering.

Output: paper/figures/fig3-pair-score-histograms.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "outputs" / "hf-runs"

PANELS = [
    ("CLIP (web)", "ham10000-hf-clip-exp006b-v1", "#9aa3b2"),
    (
        "BiomedCLIP (gen. medical)",
        "ham10000-hf-biomedclip-exp008-v1",
        "#6c95cc",
    ),
    ("DermLIP (dermatology)", "ham10000-hf-dermlip-exp007-v1", "#3d8c5c"),
]


def load_test(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    p = RUNS_ROOT / run_id / "artifacts" / "embeddings" / "jepa_predictor_latents.npz"
    data = np.load(p)
    mask = data["split"] == "test"
    return data["score"][mask], data["label"][mask]


def main() -> int:
    fig_dir = Path(__file__).resolve().parent

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)

    bins = np.linspace(0.0, 0.7, 36)

    for ax, (title, run_id, accent) in zip(axes, PANELS, strict=True):
        score, label = load_test(run_id)
        stable = score[label == "stable"]
        changing = score[label == "changing"]

        ax.hist(
            stable,
            bins=bins,
            alpha=0.55,
            color="#3d6c8c",
            label=f"stable (μ={stable.mean():.3f})",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.hist(
            changing,
            bins=bins,
            alpha=0.55,
            color="#c76b3d",
            label=f"changing (μ={changing.mean():.3f})",
            edgecolor="white",
            linewidth=0.4,
        )

        # Mean lines.
        ax.axvline(
            stable.mean(),
            color="#3d6c8c",
            linestyle="--",
            linewidth=0.7,
        )
        ax.axvline(
            changing.mean(),
            color="#c76b3d",
            linestyle="--",
            linewidth=0.7,
        )

        # Inversion annotation: arrow from stable mean to changing mean.
        oriented = stable.mean() < changing.mean()
        tag = "correctly oriented" if oriented else "inverted (stable > changing)"
        ax.set_title(f"{title}\n{tag}", fontsize=9, color=accent)
        ax.set_xlabel("Cosine distance (predicted, target)")
        ax.legend(fontsize=7.5, loc="upper right", frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Pair count (test split, $n=2{,}000$)")

    fig.suptitle(
        "Test-split pair-score histograms: stable vs changing "
        "under three pretraining domains",
        fontsize=10,
        y=1.04,
    )
    fig.tight_layout()

    out = fig_dir / "fig3-pair-score-histograms.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
