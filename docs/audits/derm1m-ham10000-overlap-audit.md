# Derm1M ↔ HAM10000 image-overlap audit

**Date:** 2026-06-05
**Script:** [`scripts/audit_derm1m_overlap.py`](../../scripts/audit_derm1m_overlap.py)
**Question:** Did DermLIP's pretraining corpus (Derm1M, arXiv:2503.14911) contain
HAM10000 images? This is the central caveat on the EXP-007 headline (frozen
DermLIP, test AUROC 0.944), because HAM10000 image-level overlap and
dermoscopy-domain transfer are confounded by it.

## TL;DR

HAM10000 is **not** a named Derm1M source and is **absent from Derm1M's
ISIC-labeled partition** (0 of 10,015 HAM IDs). But a perceptual-hash audit
**confirms HAM10000 images did leak into Derm1M** via reproduced figures in its
scraped PubMed/literature partition: **at least 13 distinct HAM10000 dermoscopy
images, drawn from ≥4 journal articles**, verified by agreement across four
independent perceptual hashes and by visual inspection. The overlap is real but
small-scale and indirect (literature scraping, not dataset inclusion), and 13 is
a **lower bound** — the forum/textbook channels were only partly verified and the
16.6 GB YouTube-frame partition was not audited. The practical consequence is
unchanged: DermLIP's pretraining was not HAM10000-free, so EXP-007's attribution
(transfer vs overlap) stays confounded and EXP-009 remains the partition.

## Method

Three checks, cheap to expensive:

1. **Exact-ID** — HAM10000 images carry canonical `ISIC_XXXXXXX` IDs. Derm1M's
   released manifest (`Derm1M_v2_pretrain.csv`, 413,210 rows) retains original
   filenames, so this is meaningful. Result: **0** of HAM10000's 10,015 IDs
   appear.
2. **ID-range** — Derm1M carries 14,364 ISIC-archive images (IDs 9,868–9,999,251);
   **0** fall in HAM10000's contiguous block (24,306–34,320). Derm1M's ISIC
   content is a different ISIC subset (consistent with the named MSKCC source
   being a different contributing institution than HAM10000's Vienna/Rosendahl
   images).
3. **Perceptual hash** — pHash of all 10,015 HAM images against every image in
   the source-partitioned zips (`public, pubmed, IIYI, reddit, twitter, note,
   edu`; **YouTube excluded**), then strict re-verification of low-distance
   candidates with three further hashes (dHash, aHash, wHash) and visual
   inspection.

**False-positive calibration (important).** Dermoscopy images are visually
homogeneous (centered lesion, similar scope vignette), so pHash collides
heavily: the `public/ISIC_*` and `public/dermoscopy-…-isic20XX-task1-…` files
have *known non-HAM* IDs yet produced 21,099 "near-duplicates" to HAM images at
Hamming distance ≤10. Raw pHash distance is therefore unusable as a duplicate
test here; only multi-hash agreement + visual checks were trusted.

## Coverage

| Partition | Images hashed | Audited |
|---|---:|:--:|
| public | 40,974 | ✓ |
| pubmed | 77,552 | ✓ |
| IIYI (forum) | 53,747 | ✓ (candidates not all hand-verified) |
| edu (textbooks) | 27,151 | ✓ (candidates not all hand-verified) |
| note | 12,506 | ✓ |
| twitter | 6,116 | ✓ |
| reddit | 1,173 | ✓ |
| **youtube** | ~193,991 | **✗ not audited (16.6 GB)** |

## Findings

Confirmed distinct HAM10000 images present in Derm1M's PubMed/`public` literature
partition (≥3 of 4 hashes within Hamming 6, visually verified for the distance-0
pairs): **13**, from ≥4 PubMed Central articles.

- Articles: PMC8391467, PMC8997449, PMC9316548, PMC9777332.
- HAM IDs: ISIC_0024562, ISIC_0024826, ISIC_0025424, ISIC_0025831,
  ISIC_0026650, ISIC_0027006, ISIC_0027198, ISIC_0027293, ISIC_0027675,
  ISIC_0027903, ISIC_0027904, ISIC_0028651, ISIC_0031293.
- Example: a single figure (`PMC9777332`, *Diagnostics* 12:03145, Fig. 2) is a
  multi-panel montage reproducing ~5 distinct HAM10000 dermoscopy images; each
  panel matched a different HAM image at phash/dhash/ahash/whash ≈ 0. Two
  distance-0 pairs (`ISIC_0025424`↔`PMC9777332…g002_5`,
  `ISIC_0024562`↔`PMC8997449…f009`) were confirmed identical by eye.

## Limitations

- **Lower bound.** pHash only catches images that survive as recognizable
  figures; YouTube frames were not audited; forum/textbook candidates were not
  all individually verified; and the gated v2 Hub release "differs slightly"
  from the trained checkpoint per the Derm1M README, so the audited image set is
  not guaranteed identical to DermLIP's training set.
- **Access.** Derm1M is gated; this audit required granted access. A third party
  cannot reproduce it without the same grant.

## Implication for EXP-007

DermLIP saw at least some HAM10000 images during pretraining, so the EXP-007
positive cannot be cleanly attributed to dermoscopy-domain transfer. The
confirmed scale is small (≥13 of 10,015, a lower bound), which makes wholesale
memorization an unlikely *sole* explanation but does not clear the confound. The
EXP-009 partition (self-pretrain on a non-HAM10000 dermoscopy corpus) remains the
way to separate transfer from overlap.
