# DermaJEPA experiment reports

One Markdown file per primary-tier run. Each report is self-contained: you
should be able to cite numbers from it in a blog post, talk, or paper
without re-opening the raw run directory.

## Report format

Every `EXP-<id>-<slug>.md` file follows the same section skeleton so future
runs are directly comparable and so they aggregate cleanly into future
writeups:

1. **Summary** — one-paragraph plain-English take and the three key numbers.
2. **Experimental setup** — config, dataset, splits, hardware, environment.
3. **Operational timeline** — wall-clock phases, costs, observability notes.
4. **Results** — headline metrics table, baseline comparison, CI bands.
5. **Analysis** — what the numbers mean, why they came out that way.
6. **Failure cases** — worst stable and worst changing pairs.
7. **Limitations and threats to validity** — the honest section.
8. **What changes for the next run** — concrete, bounded follow-ups.
9. **Reproducibility** — exact commands, config hash, run location.
10. **Assets for future writeups** — pre-extracted numbers, plot paths,
    quotable sentences.

Reports cite the run's uploaded Hub location so a reader can independently
reproduce every number.

## Index

| ID | Run ID | Tier | Date (UTC) | Primary AUROC | Strongest baseline | Δ | Outcome |
|---|---|---|---|---|---|---|---|
| [EXP-001](EXP-001-ham10000-jepa-primary-v1.md) | `ham10000-hf-dinov2-primary-v1` | public | 2026-04-22 | 0.9998 | DINOv2 ViT-S/14 cos = 1.0000 | -0.0001 | Ceiling / proxy-construction-bound |
| [EXP-002](EXP-002-ham10000-jepa-hardened-proxy-v1.md) | `ham10000-hf-dinov2-exp002-v1` | public | 2026-04-22 | 0.9201 [0.9084, 0.9313] | DINOv2 ViT-S/14 cos = 0.6515 [0.6272, 0.6744] | **+0.2687** | Positive, non-overlapping CIs |
| [EXP-003](EXP-003-ham10000-jepa-held-out-nuisance-v1.md) | `ham10000-hf-dinov2-exp003-v1` | public | 2026-04-23 | 0.6795 [0.6563, 0.7021] | SSIM distance = 0.9605 [0.9529, 0.9677] | **−0.2810** | Falsification positive: EXP-002 delta does not generalize to held-out nuisance family |
