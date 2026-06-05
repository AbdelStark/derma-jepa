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


def _phash_dir(images: list[Path]) -> dict[str, str]:
    import imagehash
    from PIL import Image
    from tqdm import tqdm

    out: dict[str, str] = {}
    for p in tqdm(images, desc="phash"):
        try:
            with Image.open(p) as im:
                out[p.name] = str(imagehash.phash(im.convert("RGB")))
        except Exception:  # noqa: BLE001 - skip unreadable images, keep auditing
            continue
    return out


def stage_phash(derm1m_zip: str, threshold: int) -> None:
    """Near-duplicate overlap via perceptual hashing (re-encoding robust)."""
    import imagehash
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        ham_dir = Path(
            snapshot_download(HAM_REPO, repo_type="dataset", allow_patterns="*.jpg")
        )
        zpath = hf_hub_download(
            DERM1M_REPO, f"{derm1m_zip}.zip", repo_type="dataset"
        )
    except GatedRepoError:
        sys.exit(
            "BLOCKED: Derm1M is gated. Request access first, then re-run."
        )

    derm_dir = OUT / f"derm1m_{derm1m_zip}"
    derm_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(derm_dir)

    ham = _phash_dir(sorted(ham_dir.rglob("*.jpg")))
    derm = _phash_dir(
        sorted(p for p in derm_dir.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    )
    print(f"HAM images hashed: {len(ham)}  Derm1M[{derm1m_zip}] hashed: {len(derm)}")

    derm_hashes = {k: imagehash.hex_to_hash(v) for k, v in derm.items()}
    near = []
    for hname, hhex in ham.items():
        hh = imagehash.hex_to_hash(hhex)
        for dname, dh in derm_hashes.items():
            if (hh - dh) <= threshold:
                near.append((hname, dname, hh - dh))
    near.sort(key=lambda t: t[2])

    report = OUT / f"phash_overlap_{derm1m_zip}.csv"
    with open(report, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ham_image", "derm1m_image", "hamming_distance"])
        w.writerows(near)
    print(f"near-duplicate pairs (<= {threshold}): {len(near)}  -> {report}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("id", "phash"), default="id")
    ap.add_argument(
        "--derm1m-zip",
        default="public",
        help="which Derm1M source zip to hash (public/pubmed/youtube/...)",
    )
    ap.add_argument("--threshold", type=int, default=6, help="max pHash Hamming dist")
    args = ap.parse_args()
    if args.stage == "id":
        stage_id()
    else:
        stage_phash(args.derm1m_zip, args.threshold)


if __name__ == "__main__":
    main()
