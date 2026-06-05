#!/usr/bin/env python3
"""Audit HAM10000 <-> Derm1M image overlap.

Background
---------
The DermLIP backbone behind EXP-007 is CLIP-trained on Derm1M
(arXiv:2503.14911). Derm1M does not name HAM10000 as an image source (its
named public datasets are SCIN and MSKCC), but it documents no decontamination
against HAM10000 and bulk-aggregates literature/web/social sources where
HAM10000 images circulate. This script tries to turn "overlap cannot be
excluded" into a measured answer, at two levels:

  1. Exact-ID check  (cheap): HAM10000 images carry canonical ISIC_XXXXXXX
     IDs. If Derm1M's released manifest (Derm1M_v2_pretrain.csv) keeps original
     filenames, grep it for HAM10000's exact IDs.
  2. Perceptual-hash check (heavier): pHash every HAM10000 image and every
     Derm1M image, then report near-duplicate pairs under a Hamming-distance
     threshold. This catches re-encoded / re-captioned copies that an exact-ID
     match would miss.

Access
------
`redlessone/Derm1M` is a GATED dataset. You must request access on the Hub
("Agree and access") before this script can read the manifest or images;
otherwise the downloads raise GatedRepoError (403). The gating is itself part
of why third-party overlap auditing is not trivial.

Usage
-----
    uv run --with "huggingface-hub>=1.0,pandas,pyarrow,imagehash,pillow,tqdm" \
        python scripts/audit_derm1m_overlap.py --stage id
    # then, once you have decided to download the image zips:
    uv run ... python scripts/audit_derm1m_overlap.py --stage phash --derm1m-zip public

The script writes findings to outputs/audits/derm1m_overlap/.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

HAM_REPO = "abdelstark/ham10000"  # our private HAM10000 mirror
DERM1M_REPO = "redlessone/Derm1M"  # gated
OUT = Path("outputs/audits/derm1m_overlap")


def ham10000_isic_ids() -> set[str]:
    """The 10,015 HAM10000 image IDs (ISIC_XXXXXXX), from the mirror's filenames."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(HAM_REPO, repo_type="dataset")
    ids = {
        Path(f).stem
        for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and "ISIC_" in f
    }
    if not ids:
        sys.exit(f"no HAM10000 image IDs found in {HAM_REPO}; check access")
    return ids


def stage_id() -> None:
    """Exact-ID overlap: does the Derm1M manifest contain HAM10000 ISIC IDs?"""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    OUT.mkdir(parents=True, exist_ok=True)
    ids = ham10000_isic_ids()
    print(f"HAM10000 IDs: {len(ids)} (e.g. {sorted(ids)[:3]})")

    try:
        manifest = hf_hub_download(
            DERM1M_REPO, "Derm1M_v2_pretrain.csv", repo_type="dataset"
        )
    except GatedRepoError:
        sys.exit(
            "BLOCKED: Derm1M is gated. Request access at "
            "https://huggingface.co/datasets/redlessone/Derm1M "
            "('Agree and access'), then re-run."
        )

    hits, n_rows = [], 0
    with open(manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        key = next((c for c in (reader.fieldnames or []) if "file" in c.lower()), None)
        if key is None:
            sys.exit(f"no filename column in manifest; columns={reader.fieldnames}")
        for row in reader:
            n_rows += 1
            stem = Path(str(row[key])).stem
            if stem in ids:
                hits.append(row[key])

    report = OUT / "id_overlap.txt"
    report.write_text(
        f"Derm1M manifest rows: {n_rows}\n"
        f"HAM10000 IDs checked: {len(ids)}\n"
        f"exact-ID hits: {len(hits)}\n" + "\n".join(hits)
    )
    print(f"manifest rows={n_rows}  exact-ID hits={len(hits)}  -> {report}")
    if hits:
        print("OVERLAP CONFIRMED at the ID level (see report).")
    else:
        print(
            "No exact-ID hits in the released manifest. This does NOT prove "
            "absence: filenames may be anonymised, the v2 release may differ "
            "from the trained checkpoint, and re-encoded copies need the phash "
            "stage."
        )


_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_POPCOUNT = None


def _hash_to_u64(ih: object) -> int:
    """Pack an 8x8 pHash (imagehash.ImageHash) into a uint64."""
    v = 0
    for bit in ih.hash.flatten():  # type: ignore[attr-defined]
        v = (v << 1) | int(bit)
    return v


def _popcounts(xs):
    """Vectorised popcount of a uint64 numpy array via an 8-bit lookup table."""
    import numpy as np

    global _POPCOUNT
    if _POPCOUNT is None:
        _POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return _POPCOUNT[xs.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def _ham_phashes() -> tuple[list[str], object]:
    """pHash every HAM10000 image once; return (ids, uint64 array)."""
    import imagehash
    import numpy as np
    from huggingface_hub import snapshot_download
    from PIL import Image
    from tqdm import tqdm

    ham_dir = Path(
        snapshot_download(HAM_REPO, repo_type="dataset", allow_patterns="*.jpg")
    )
    ids, vals = [], []
    for p in tqdm(sorted(ham_dir.rglob("*.jpg")), desc="phash:HAM10000"):
        try:
            with Image.open(p) as im:
                vals.append(_hash_to_u64(imagehash.phash(im.convert("RGB"))))
                ids.append(p.stem)
        except Exception:  # noqa: BLE001 - skip unreadable, keep auditing
            continue
    return ids, np.array(vals, dtype=np.uint64)


def stage_phash(zips: list[str], threshold: int) -> None:
    """Near-duplicate overlap via perceptual hashing (re-encoding robust).

    Streams each Derm1M source zip member-by-member (no full disk extract) and
    compares every image's pHash against the HAM10000 set. Reports pairs within
    `threshold` Hamming distance, per source zip and in total.
    """
    import imagehash
    import numpy as np
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    from PIL import Image
    from tqdm import tqdm

    OUT.mkdir(parents=True, exist_ok=True)
    ham_ids, ham_arr = _ham_phashes()
    print(f"HAM10000 images hashed: {len(ham_ids)}")

    all_pairs: list[tuple[str, str, str, int]] = []
    coverage: list[tuple[str, int]] = []
    for z in zips:
        try:
            zpath = hf_hub_download(DERM1M_REPO, f"{z}.zip", repo_type="dataset")
        except GatedRepoError:
            sys.exit("BLOCKED: Derm1M is gated. Request access first, then re-run.")
        n_hashed, pairs = 0, []
        with zipfile.ZipFile(zpath) as zf:
            members = [m for m in zf.namelist() if Path(m).suffix.lower() in _IMG_EXT]
            for name in tqdm(members, desc=f"phash:{z}"):
                try:
                    with zf.open(name) as fh, Image.open(fh) as im:
                        dh = _hash_to_u64(imagehash.phash(im.convert("RGB")))
                except Exception:  # noqa: BLE001 - skip unreadable, keep auditing
                    continue
                n_hashed += 1
                dists = _popcounts(ham_arr ^ np.uint64(dh))
                for i in np.where(dists <= threshold)[0]:
                    d = int(dists[i])
                    pairs.append((ham_ids[i], f"{z}/{name}", str(d), d))
        coverage.append((z, n_hashed))
        all_pairs.extend(pairs)
        print(f"  {z}: hashed {n_hashed}  near-dups (<= {threshold}): {len(pairs)}")
        Path(zpath).unlink(missing_ok=True)  # free disk between zips

    all_pairs.sort(key=lambda t: t[3])
    report = OUT / "phash_overlap.csv"
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ham_image", "derm1m_image", "hamming_distance"])
        w.writerows((a, b, c) for a, b, c, _ in all_pairs)
    cov = OUT / "phash_coverage.txt"
    cov.write_text(
        f"threshold={threshold}\nHAM10000 hashed={len(ham_ids)}\n"
        + "\n".join(f"{z}: {n} images hashed" for z, n in coverage)
        + f"\nTOTAL near-duplicate pairs: {len(all_pairs)}\n"
    )
    bands = {b: sum(1 for *_, d in all_pairs if d <= b) for b in (0, 2, 4, 6, 8, 10)}
    print(f"\nTOTAL near-dups by distance band: {bands}")
    print(f"coverage -> {cov}\npairs -> {report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("id", "phash"), default="id")
    ap.add_argument(
        "--zips",
        default="public,pubmed,IIYI,reddit,twitter,note,edu",
        help="comma-separated Derm1M source zips to hash (omit 'youtube' for speed)",
    )
    ap.add_argument("--threshold", type=int, default=10, help="max pHash Hamming dist")
    args = ap.parse_args()
    if args.stage == "id":
        stage_id()
    else:
        zips = [z.strip() for z in args.zips.split(",") if z.strip()]
        stage_phash(zips, args.threshold)


if __name__ == "__main__":
    main()
