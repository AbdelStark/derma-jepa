# Data

DermaJEPA does not vendor public dermatology datasets or raw images. Keep raw
downloads under `data/raw/`, which is gitignored, and regenerate manifests and
run artifacts from checked-in configs.

## HAM10000 Layout

The Milestone 2 public-data path expects a HAM10000-style local layout:

```text
data/raw/ham10000/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
    ISIC_*.jpg
  HAM10000_images_part_2/
    ISIC_*.jpg
```

The default config also accepts a flat `data/raw/ham10000/images/` directory.

### Hugging Face Jobs mounted layout

For hosted runs, mount a Hub dataset repo at `/data` and use
`configs/data/ham10000_hf_mounted.yaml`. The mounted repo must expose:

```text
/data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
    ISIC_*.jpg
  HAM10000_images_part_2/
    ISIC_*.jpg
```

`/data/images/` is also accepted as a flat fallback. Launch with, for example:

```bash
HF_JOBS_VOLUME="hf://datasets/<namespace>/<ham10000-repo>:/data:ro" \
DERMA_JEPA_CONFIG_PATH=configs/data/ham10000_hf_mounted.yaml \
./scripts/hf_jobs_train_bundle.sh
```

Do not upload raw patient or personal images to any public Hub repo.

Required metadata columns:

- `image_id`
- `lesion_id`
- `dx`
- `localization`

Optional metadata columns:

- `patient_id` or equivalent patient/subject identifier
- `dx_type`
- `age`
- `sex`

If patient identifiers are absent, the splitter uses lesion IDs as the leakage
boundary. Rows without lesion IDs are allowed only as exploratory image-level
fallbacks and are called out in `data_audit.json`.

## Source, Access, And Citation

HAM10000 is the primary MVP-scale dermoscopic source because it is widely used
and includes lesion IDs. Download it from the official dataset host available to
the project operator and review the current access terms before use. Do not
commit images, raw metadata if redistribution is restricted, credentials, or
derived artifacts that include raw images.

Citation text to carry into reports when HAM10000 is used:

> Tschandl, P., Rosendahl, C., and Kittler, H. The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions. Scientific Data, 2018.

## Commands

Audit local metadata and image availability:

```bash
uv run derma-jepa data audit --config configs/data/ham10000.yaml
```

Build the leakage-aware longitudinal-proxy manifest:

```bash
uv run derma-jepa manifest build --config configs/data/ham10000.yaml
```

Run cheap public-tier baselines on the held-out split:

```bash
uv run derma-jepa embed --config configs/data/ham10000.yaml
uv run derma-jepa baseline eval --config configs/data/ham10000.yaml
```

The default HAM10000 config exports DINOv2 ViT-S/14 and ViT-B/14 embeddings.
Install optional model dependencies first:

```bash
uv sync --extra model
```

Generated run files live under `runs/ham10000-proxy-v1/`:

- `metadata_normalized.parquet`
- `data_audit.json`
- `artifacts/reports/gold_audit_subset.csv`
- `manifest_all.parquet`
- `manifest_train.parquet`
- `manifest_val.parquet`
- `manifest_test.parquet`
- `artifacts/embeddings/embedding_index.json`
- `artifacts/embeddings/dinov2_vits14.npz`
- `artifacts/embeddings/dinov2_vitb14.npz`
- `baseline_metrics.json`
- `metrics.json`
- `model_card.md`
- `artifacts/plots/baseline_score_histogram.png`
- `artifacts/reports/baseline_failure_cases.json`

## Leakage Rules

The manifest builder enforces the shared manifest contract:

- train/validation/test split groups are patient IDs when available
- lesion IDs are the fallback split groups for HAM10000-style metadata
- stable pairs are generated after split assignment from mild nuisance variants
- changing pairs require different lesion IDs and prefer same-patient, then
  same-diagnosis/site, then same-diagnosis, then same-site matches
- duplicate `image_id` values stop manifest generation
- missing local images are reported by audit and stop manifest generation

The public-data path is still a longitudinal-proxy task. It is a research demo,
not diagnostic, not medical advice, and not validated for patient use.
