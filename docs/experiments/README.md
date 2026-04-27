# DermaJEPA experiment reports

One Markdown file per primary-tier run. Each report is self-contained: the numbers in it are quotable in a paper, talk, or blog post without re-opening the raw run directory.

## Report format

Every `EXP-<id>-<slug>.md` file follows the same section skeleton so future runs are directly comparable and aggregate cleanly into future writeups:

1. **Summary** — one-paragraph plain-English take and the headline numbers.
2. **Experimental setup** — config, dataset, splits, hardware, environment.
3. **Operational timeline** — wall-clock phases, costs, observability notes.
4. **Results** — headline metrics table, baseline comparison, CI bands.
5. **Analysis** — what the numbers mean, why they came out that way.
6. **Limitations and threats to validity** — the honest section.
7. **What changes for the next run** — concrete, bounded follow-ups.
8. **Reproducibility** — exact commands, config diff, run location.
9. **Assets for future writeups** — pre-extracted numbers, plot paths, quotable sentences, paper-section mapping.
10. **Changelog** — date / author / what was added or revised.

Reports cite the run's uploaded Hub location so a reader can independently reproduce every number.

## Index

| ID | Run ID | Tier | Date (UTC) | Primary AUROC | Strongest baseline | Δ | Outcome |
|---|---|---|---|---|---|---|---|
| [EXP-001](EXP-001-ham10000-jepa-primary-v1.md) | `ham10000-hf-dinov2-primary-v1` | public | 2026-04-22 | 0.9998 | DINOv2 ViT-S/14 cos = 1.0000 | -0.0001 | Ceiling / proxy-construction-bound |
| [EXP-002](EXP-002-ham10000-jepa-hardened-proxy-v1.md) | `ham10000-hf-dinov2-exp002-v1` | public | 2026-04-22 | 0.9201 [0.9084, 0.9313] | DINOv2 ViT-S/14 cos = 0.6515 [0.6272, 0.6744] | **+0.2687** | Positive, non-overlapping CIs |
| [EXP-003](EXP-003-ham10000-jepa-held-out-nuisance-v1.md) | `ham10000-hf-dinov2-exp003-v1` | public | 2026-04-23 | 0.6795 [0.6563, 0.7021] | SSIM distance = 0.9605 [0.9529, 0.9677] | **−0.2810** | Falsification: EXP-002 delta does not generalise to held-out nuisance family |
| [EXP-004](EXP-004-ham10000-jepa-mixed-train-held-out-2-v1.md) | `ham10000-hf-dinov2-exp004-v1` | public | 2026-04-23 | 0.2491 [0.2296, 0.2698] | Pixel L2 = 0.5802 [0.5560, 0.6058] | **−0.3311** | Below-random + inverted; mixed-family training does not rescue generalisation; linear-over-frozen-DINOv2 identified as bottleneck |
| [EXP-005](EXP-005-ham10000-jepa-mlp-predictor-v1.md) | `ham10000-hf-dinov2-exp005-v1` | public | 2026-04-24 | 0.2702 [0.2491, 0.2926] | Pixel L2 = 0.5802 | **−0.3101** | MLP predictor underfit under inherited optimiser (train AUROC 0.572); test AUROC ≈ DINOv2 baseline. Capacity hypothesis not cleanly tested. |
| [EXP-006a](EXP-006a-ham10000-jepa-adam-mlp-v1.md) | `ham10000-hf-dinov2-exp006a-v1` | public | 2026-04-24 | 0.2480 [0.2279, 0.2691] | Pixel L2 = 0.5802 | **−0.3322** | Adam-tuned MLP fits training (AUROC 0.893) but produces same below-random test as EXP-004 linear. Scaffold-capacity hypothesis not supported. |
| [EXP-006b](EXP-006b-ham10000-jepa-clip-backbone-v1.md) | `ham10000-hf-clip-exp006b-v1` | public | 2026-04-24 | 0.2864 [0.2654, 0.3097] | Pixel L2 = 0.5802 | **−0.2939** | Same below-random inversion under frozen OpenAI CLIP ViT-B/16. Failure is not DINOv2-specific. |
| [EXP-007](EXP-007-ham10000-jepa-dermlip-backbone-v1.md) | `ham10000-hf-dermlip-exp007-v1` | public | 2026-04-27 | 0.9447 [0.9346, 0.9537] | Pixel L2 = 0.5802 | **+0.3645** | Frozen DermLIP (dermoscopy-CLIP) lifts test AUROC to 0.945. First above-random result on `strong_held_out_2` since EXP-002. Pretraining-contamination caveat: Derm1M almost certainly includes HAM10000. |
| [EXP-008](EXP-008-ham10000-jepa-biomedclip-backbone-v1.md) | `ham10000-hf-biomedclip-exp008-v1` | public | 2026-04-27 | 0.3247 [0.3026, 0.3493] | Pixel L2 = 0.5802 | **−0.2556** | Frozen BiomedCLIP (general-medical, no HAM10000) lifts only +0.04 over web CLIP. Partition: EXP-007's win is concentrated at the dermoscopy-specific pretraining step, not "any medical pretraining." |
| [EXP-007/008 seed sweep](EXP-007-008-seed-sweep-summary.md) | 5 seeds × 2 configs | public | 2026-04-27 | DermLIP 0.944 ± 0.003 / BiomedCLIP 0.329 ± 0.012 | — | — | Both headlines are seed-stable. Three-way pretraining-data gradient (web < general-medical < dermoscopy) holds with seed-mean point estimates. |
