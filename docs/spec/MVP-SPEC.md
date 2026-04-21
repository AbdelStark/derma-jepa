# DermaJEPA MVP specification

Status: Accepted
Last updated: 2026-04-21

This document is the canonical MVP implementation contract. The RFCs record the
same decisions in smaller decision records, but implementation should treat this
file as the source of truth when there is any ambiguity.

## 1. MVP claim and non-claims

The MVP claim is:

> DermaJEPA evaluates whether a JEPA-style latent trajectory model can separate
> nuisance-induced visual drift from meaningful lesion-change proxies better than
> simple pixel/SSIM baselines and generic frozen vision embeddings on public
> dermatology data.

Allowed positive-result wording:

> On a leakage-controlled longitudinal-proxy task, the JEPA-style drift score
> improved over the strongest baseline by X AUROC points with bootstrap
> confidence interval [a, b].

Allowed negative-result wording:

> On the locked proxy task, the JEPA-style model did not outperform the strongest
> baseline; the MVP is therefore an analytical research artifact rather than
> evidence for the thesis.

The MVP must not claim:

- diagnostic accuracy
- clinical validity
- melanoma detection
- cancer-risk prediction
- treatment recommendation
- real patient monitoring
- medical-device readiness
- full JEPA pretraining from scratch
- a general dermatology foundation model

The project may use a world-model framing only in this restricted sense:

> The system learns a latent predictive model over lesion-image representations.
> Prediction happens in representation space, over image-derived lesion states,
> under a controlled proxy task.

Avoid language such as "understands disease progression", "models melanoma
evolution", "clinical world model", "patient simulator", or "foundation model
for dermatology".

## 2. Target audience and credibility bar

The primary audience is academic ML scientists, ML researchers, and world-model
researchers evaluating whether the JEPA framing is technically meaningful for
lesion monitoring. The secondary audience is future contributors who need a
reproducible research engineering surface rather than a notebook demo.

Credibility requires:

- a locked proxy task before model work
- leakage-aware manifests
- strong cheap baselines before the JEPA-style model is interpreted
- confidence intervals for the primary result
- visible failure cases
- complete run artifacts
- explicit separation between metric results, robustness results, runtime
  results, and qualitative cases
- safety language that makes clear this is not a medical product

## 3. Research and hardware grounding

The MVP is constrained by public data and local/desktop compute.

Relevant source facts as of 2026-04-21:

- I-JEPA predicts latent target representations from image context rather than
  reconstructing pixels. The official I-JEPA repository states that the
  ViT-H/14 reproduction config should run on 16 A100 80GB GPUs for effective
  batch size 2048. The I-JEPA paper reports training ViT-Huge/14 on ImageNet
  using 16 A100 GPUs in under 72 hours.
- DINOv2 provides strong frozen vision baselines with published ViT-S/14
  21M-parameter and ViT-B/14 86M-parameter checkpoints.
- HAM10000 contains 10,015 dermoscopic images and is a familiar public
  dermatology benchmark.
- PAD-UFES-20 contains 2,298 smartphone/clinical images from 1,641 lesions and
  1,373 patients. It is useful as an external stress/domain-shift test, not the
  primary training source.
- ISIC Archive is a large public skin-image archive with evolving metadata and
  command-line access. It can be used for metadata audits and extended dataset
  selection, but the MVP must not depend on discovering a perfect public
  longitudinal subset.
- GB10/DGX Spark-class systems provide 128GB coherent unified memory and a
  CUDA/PyTorch-friendly software stack, but the advertised FP4 inference
  performance does not make full I-JEPA reproduction an MVP target.

Source links:

- I-JEPA repository: https://github.com/facebookresearch/ijepa
- I-JEPA paper: https://arxiv.org/abs/2301.08243
- DINOv2 repository/model table: https://github.com/facebookresearch/dinov2
- NVIDIA DGX Spark specifications: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- Apple MacBook Pro technical specs: https://support.apple.com/en-us/121554
- ISIC Archive: https://www.isic-archive.com/
- HAM10000 paper: https://www.nature.com/articles/sdata2018161
- PAD-UFES-20 paper: https://arxiv.org/abs/2007.00478

## 4. Dataset policy and local data layout

The repo must not vendor public dermatology datasets or raw images.

Dataset priority:

1. Primary MVP source: ISIC/HAM10000-style dermoscopic images for scale and
   research familiarity.
2. External stress source: PAD-UFES-20 for smartphone/clinical-image
   domain-shift reporting.
3. Optional extension: selected ISIC Archive subsets after metadata and leakage
   audits show usable grouping.

Target scale:

- Fixture tier: 20-100 synthetic or license-safe images for CI and local smoke
  tests.
- Gold audit tier: 100-300 manually inspected public images/pairs for visual
  sanity checks and failure-case templates.
- Primary MVP tier: HAM10000-scale run, roughly 10k dermoscopic images, with
  generated stable/changing proxy pairs.
- Extended ISIC tier: 25k-50k selected ISIC images only after the HAM10000-scale
  pipeline works.
- External stress tier: full PAD-UFES-20, around 2.3k smartphone/clinical
  images, used for robustness and domain-shift reporting.

Local layout:

- `data/raw/`: gitignored raw downloads.
- `data/interim/`: gitignored extracted metadata and resized caches.
- `data/manifests/`: versionable only when metadata license allows; otherwise
  gitignored with a reproducible generation command.
- `runs/`: gitignored run outputs by default.
- `reports/`: selected small reports, figures, and model cards that are safe to
  commit.
- `tests/fixtures/`: synthetic or license-safe tiny images only.

Data scripts must document the source, license or access terms, expected files,
checksums where possible, and citation text. Demo cases must come from public or
research-allowed sources and carry source attribution in run provenance. User
personal photos and patient data are out of scope for MVP artifacts.

## 5. Stable/changing proxy task

The MVP evaluates a longitudinal-proxy change-detection task. It is not real
clinical progression detection.

Stable pairs:

- the same source lesion image under strong nuisance perturbations
- same-lesion duplicate images when lesion identifiers make this reliable

Changing pairs:

- different lesion images matched to reduce trivial shortcuts
- priority order for matching:
  - same patient when available
  - same anatomical site when available
  - same diagnosis class when available
  - visually similar negatives from nearest-neighbor mining

Changing pairs must not be treated as biological progression. They are proxy
departures from lesion identity or morphology under matched confounders.

The pair/window manifest must include:

- original image IDs
- pair or window ID
- split
- source dataset
- pair label: `stable` or `changing`
- diagnosis if available
- patient ID if available
- lesion ID if available
- anatomical site if available
- source/device metadata if available
- preprocessing profile
- augmentation recipe and parameters
- checksum for raw image references where possible
- reason code for pair construction

## 6. Split and leakage policy

The leakage policy is strict by default:

- No image from the same lesion may appear across train/val/test when lesion IDs
  exist.
- No patient may appear across train/val/test when patient IDs exist.
- Stable augmented pairs must be generated after the split, never before.
- If only diagnosis labels exist, split by source archive and nearest-neighbor
  duplicate clusters to reduce leakage risk.
- Source-aware reporting is required when datasets come from multiple archives,
  devices, or collection sites.
- Any result that cannot satisfy the split policy must be marked exploratory,
  not main MVP evidence.

Leakage probes are mandatory:

- patient/lesion overlap checks where metadata exists
- duplicate or near-duplicate cluster checks
- source/dataset shortcut checks
- diagnosis-class shortcut checks for changing pairs
- image-size or preprocessing shortcut checks where raw metadata exposes them

The data audit must end with a short leakage-risk note before full training.

## 7. Image preprocessing contract

Preprocessing must be standardized before model work:

- Decode RGB only.
- Reject unreadable or corrupt files with manifest-level error records.
- Preserve original image path, dimensions, source dataset, and checksum.
- Generate normalized model input at `224x224` for DINOv2-S/B and first
  JEPA-style predictor runs.
- Use resize-shortest-side plus center crop unless source metadata or masks
  support reproducible lesion-aware cropping.
- Keep optional `336x336` or `448x448` profiles for later ablations, not first
  MVP.
- Use ImageNet normalization for pretrained backbones unless a backbone requires
  otherwise.
- Never overwrite raw images.
- Store resized or cached tensors/images under `data/interim/` or a run-local
  cache.
- Record a named preprocessing profile in every manifest row, for example
  `dinov2_224_center_crop_v1`.

Manual lesion cropping is not allowed in the main MVP path unless the operation
is reproducible and recorded.

## 8. Nuisance augmentation suite

The MVP uses a fixed, named nuisance suite with severity levels:

- brightness, contrast, and gamma jitter
- hue, saturation, and white-balance shift
- rotation and mild perspective warp
- scale, crop, and translation
- Gaussian blur and motion blur
- JPEG compression
- sensor noise
- synthetic occlusion overlays:
  - hair-like thin lines
  - ruler/marker-like edge occlusions
  - small random masks

Rules:

- Stable training pairs sample mild-to-moderate severities.
- Robustness tests include mild, moderate, and severe levels.
- Severe augmentations are reported separately and cannot define the main
  success result.
- Augmentation parameters must be written into the manifest for every generated
  pair.
- Augmentations must not change diagnosis labels or pretend to simulate
  biological lesion progression.

## 9. Model stack

The MVP model path is a JEPA-style adaptation, not full JEPA pretraining.

Mandatory baselines:

- pixel-space resized/cropped L2
- SSIM
- LPIPS if dependency cost is acceptable
- frozen DINOv2 ViT-S/14 cosine distance on class token and average patch token
- frozen DINOv2 ViT-B/14 cosine distance on class token and average patch token
- dermatology-supervised embedding baseline if labels are clean enough to train
  or fine-tune a lightweight classifier/backbone
- trivial metadata/leakage probes where metadata exists

JEPA-style MVP model:

- start from a strong frozen or lightly adapted vision foundation encoder
- train a compact projection/predictor stack over dermatology pairs/windows
- first backbone candidates:
  - DINOv2 ViT-B/14 for pragmatic feature quality and hardware fit
  - I-JEPA ViT-H/14 checkpoint for research-aligned ablation if loading and
    runtime are tractable
- treat backbone fine-tuning as optional escalation after the data/eval contract
  works
- do not attempt full I-JEPA pretraining from scratch for MVP

Embedding export contract:

- image ID
- split
- source dataset
- preprocessing profile
- model ID
- checkpoint ID
- feature type: class token, average patch token, projected latent, or predictor
  latent
- vector
- vector dtype and dimension
- inference device and batch size
- run ID

## 10. Training objective

The first objective is compact pairwise JEPA-style latent prediction:

- Encode context image or context patches with the context encoder.
- Encode target image or target patches with a frozen or momentum target encoder.
- Train a predictor/projection head to predict target latents from context
  latents.
- Use latent-space regression loss, such as cosine loss or smooth L1.
- Track variance/covariance collapse checks during training.
- For stable pairs, target is the same lesion under nuisance perturbation or a
  reliable same-lesion duplicate.
- For changing pairs, do not train the predictor to collapse them together.
  Changing pairs are primarily for evaluation.
- Optional hard-negative or ranking losses can be added only after the first
  objective works and must be ablated separately.

The first implementation must avoid full temporal transformers. Pair/window
scoring comes first. Sequence modeling is a later phase only if the pairwise
objective produces credible signal.

## 11. Metrics, evals, and benchmarks

Primary metric:

- pairwise proxy change-detection AUROC on held-out lesion/patient-aware splits

Primary score:

- latent drift score per pair/window

MVP success threshold:

- JEPA-style drift score beats the strongest cheap baseline
- minimum credible target is at least `+0.05 AUROC` over the strongest baseline
  on the primary held-out split
- report bootstrap confidence intervals, not a single number

Secondary metrics:

- AUPRC
- equal-error-rate threshold
- FPR at fixed TPR
- calibration/error curves for drift score

Nuisance robustness:

- drift distribution under each named nuisance family and severity level
- separate mild/moderate/severe reporting
- robust result must not depend only on severe synthetic artifacts

Representation health:

- embedding norm statistics
- feature variance
- covariance rank and effective rank
- nearest-neighbor duplicate audit
- collapse checks

Ablations:

- no JEPA predictor
- frozen projection versus lightly adapted projection
- context/target patch masking variants
- pair construction variants

Runtime benchmarks:

- embedding throughput
- training images per second
- eval time
- demo load time
- artifact size

The MVP report must separate metric result, robustness result, runtime result,
and qualitative case studies.

## 12. Hardware and runtime contract

The MVP has two operational tiers.

MacBook Pro local tier:

- runs the local demo
- runs inference on precomputed or lightweight on-demand embeddings
- runs small eval slices and report viewing
- does not need to train the full MVP model

Remote GB10 tier:

- runs embedding export over the full selected dataset
- runs JEPA-style predictor/projection training
- runs full benchmark/eval jobs
- produces versioned artifacts copied back to the MacBook Pro for the demo

Training budget:

- smoke training: under 30 minutes
- tiny overfit/debug run: under 1 hour
- full MVP training run: target under 12 hours, hard cap 24 hours on GB10
- any work requiring multi-A100 infrastructure is outside MVP scope

The MacBook Pro is the showcase/operator machine. GB10 is the optional
training/evaluation accelerator.

## 13. CLI command contract

The MVP must expose reproducible commands:

```bash
derma-jepa data audit --config configs/data/ham10000.yaml
derma-jepa manifest build --config configs/manifest/ham10000_proxy.yaml
derma-jepa embed --config configs/embed/dinov2_s.yaml
derma-jepa baseline eval --config configs/eval/baselines.yaml
derma-jepa train --config configs/train/jepa_predictor.yaml
derma-jepa eval --config configs/eval/jepa_predictor.yaml
derma-jepa benchmark --run runs/<run_id>
derma-jepa demo export --run runs/<run_id> --out artifacts/demo/<run_id>
derma-jepa demo --artifact artifacts/demo/<run_id>
```

Required behavior:

- Every command writes logs.
- Every command exits nonzero on contract failure.
- Every command supports `--dry-run` where useful.
- Every command prints generated artifact paths.
- Training/eval commands copy the resolved config into the run directory.
- `benchmark` fails if the acceptance gate fails unless
  `--allow-negative-result` is set for analytical negative-result reports.
- `demo` must not require raw data if exported demo artifacts exist.

Notebooks can exist for exploration, but no MVP result may require a
notebook-only step.

## 14. Run artifact contract

Every non-smoke run must produce a self-contained run directory:

```text
runs/<run_id>/
  config.yaml
  manifest_train.parquet
  manifest_val.parquet
  manifest_test.parquet
  metrics.json
  baseline_metrics.json
  model_card.md
  environment.txt
  artifacts/
    embeddings/
    plots/
    demo_cases/
  logs/
    train.log
    eval.log
```

Required files:

- `config.yaml`: full resolved config
- `manifest_train.parquet`, `manifest_val.parquet`, `manifest_test.parquet`:
  exact samples/pairs/windows
- `metrics.json`: primary and secondary metrics with confidence intervals
- `baseline_metrics.json`: all mandatory baselines
- `model_card.md`: model, data, objective, hardware, runtime, limitations
- `artifacts/embeddings/*.parquet` or `.npz`: exported embeddings with schema
- `artifacts/plots/*.png`: AUROC curves, drift histograms, embedding
  projections, nuisance robustness plots
- `artifacts/demo_cases/*.json`: deterministic cases used by the demo
- `logs/train.log` and `logs/eval.log`
- `environment.txt`: Python, PyTorch, platform, git commit, and device info

A run is not eligible for the MVP report unless this directory is complete and
reproducible from a repo command.

## 15. Demo contract

The MVP demo is a local dashboard over deterministic exported cases, not an
open-ended medical app.

Required views:

- case timeline with original images in order, source metadata, and proxy label
  hidden by default but inspectable
- latent drift chart with per-step drift score, threshold reference or
  confidence band, and baseline comparison
- embedding-space view with 2D projection or nearest-neighbor panel
- nuisance stress view with lighting/crop/angle/compression variants and drift
  scores
- failure-case view showing cases where the model is fooled or loses to a
  baseline
- run provenance panel with run ID, model ID, dataset split, metric summary,
  hardware, and commit

The demo must run from exported artifacts without live training. Optional
on-demand inference for one uploaded image pair can be added only after the
deterministic artifact path works.

## 16. Production-grade engineering requirements

Production grade for this MVP means research-operational rigor, not clinical
production infrastructure.

Required:

- Python-first package managed with `uv`
- `pyproject.toml`
- Python 3.11 or 3.12
- PyTorch and torchvision
- `timm` where useful
- Hugging Face Hub/Transformers only where it reduces model-loading friction
- Hydra or OmegaConf-style YAML configs
- Typer-based `derma-jepa` CLI
- Parquet via PyArrow/Polars or pandas, with explicit schemas
- scikit-learn metrics plus project-owned wrappers for confidence intervals and
  threshold reports
- pytest
- ruff
- mypy or pyright
- CI that runs lint, typecheck, unit tests, and the tiny fixture pipeline
- typed schemas for manifests, embeddings, metrics, and demo case JSON
- deterministic seeds where possible
- nondeterminism notes where determinism is not practical
- model-card and report generation

Not required for MVP:

- Kubernetes
- cloud deployment
- auth
- multi-user storage
- mobile app
- HIPAA production infrastructure
- real patient data handling

## 17. Safety, privacy, and clinical language policy

All public and demo copy must say this is a research monitoring demo.

Required language:

- "research demo"
- "longitudinal-proxy task"
- "latent drift"
- "not diagnostic"
- "not medical advice"
- "not validated for patient use"

Forbidden language:

- "detects melanoma"
- "predicts cancer"
- "clinical decision support"
- "diagnosis"
- "treatment recommendation"
- "safe for patient monitoring"
- "medical-device ready"

No personal photos or patient data may be used in MVP artifacts. Public dataset
images must be handled according to their license and citation requirements.

## 18. Implementation milestones

Milestone 1: contract-first fixture pipeline

- `pyproject.toml` and package scaffold
- CLI skeleton with locked command surface
- manifest schema and validation
- synthetic/tiny fixture dataset
- preprocessing profile implementation
- pixel/SSIM baseline on fixtures
- AUROC and bootstrap CI metric wrapper
- run directory writer
- demo export JSON for one fixture case
- CI test that runs fixture pipeline end to end

Acceptance:

- one command or script runs the fixture pipeline from manifest build through
  eval artifact export
- generated run directory satisfies the artifact contract for fixture tier
- tests pass locally and in CI

Milestone 2: HAM10000/ISIC data audit and baseline path

- dataset access/indexing scripts
- metadata normalization
- leakage audit
- stable/changing proxy manifest
- DINOv2-S and DINOv2-B embedding export
- pixel/SSIM and DINOv2 baseline report
- gold audit subset and failure-case templates

Milestone 3: JEPA-style predictor training

- context/target latent dataset
- predictor/projection model
- training loop
- collapse checks
- tiny overfit run
- full primary-tier training run on GB10
- run artifact export

Milestone 4: evaluation and benchmark suite

- AUROC/AUPRC/bootstrap report
- threshold and calibration report
- nuisance robustness report
- representation health report
- ablations
- runtime benchmark report

Milestone 5: local demo

- exported demo bundle
- Streamlit or lightweight local dashboard
- deterministic demo cases
- provenance panel
- failure-case view
- MacBook Pro smoke validation

Milestone 6: MVP report and package hardening

- model card
- results report
- README reproduction path
- safety language audit
- CI fixture gate
- go/no-go summary

## 19. Definition of done

MVP is complete only when all of these are true:

- data audit completed with a written leakage-risk note
- pair/window manifest generated with stable/changing labels and split
  provenance
- mandatory baselines implemented and evaluated
- JEPA-style model path trained at least once on the full MVP train split
- full eval run produces the required run directory
- primary result is either positive against the strongest baseline or explicitly
  reported as negative/inconclusive
- demo runs locally on MacBook Pro from exported artifacts
- tiny fixture pipeline runs in CI
- README explains the claim, commands, artifacts, limitations, safety boundary,
  and reproduction path
- no diagnostic or clinical language appears in demo/report copy

Non-completion conditions:

- polished UI without benchmark artifacts
- model checkpoint without leakage-controlled evaluation
- positive chart without baseline comparison
- result depending on hand-picked demo cases not present in the manifest
- hidden notebook-only processing step

## 20. Failure policy

If the JEPA-style model fails to beat the strongest baseline, the MVP still
ships as an honest negative or inconclusive research artifact.

Required behavior:

- report states that the MVP did not support the thesis on the locked proxy task
- demo shows the best model and strongest baseline side by side
- failure cases become first-class artifacts
- README avoids "improves" language and says "evaluates whether"
- next-step recommendations are analytical:
  - data proxy weakness
  - backbone choice
  - objective mismatch
  - leakage/split constraints
  - need for true longitudinal data

Threshold tuning, cherry-picked cases, and baseline removal are not allowed to
rescue the story.

## 21. Open risks and go/no-go checks

Risk: proxy task is too synthetic.

- Check: compare performance on mild nuisance pairs, visually similar changing
  negatives, and PAD-UFES-20 stress cases separately.

Risk: frozen DINOv2 already solves the task.

- Check: treat this as a legitimate baseline win and report the JEPA-style path
  as unnecessary for the locked proxy.

Risk: model learns dataset/source shortcuts.

- Check: source-aware splits, metadata probes, and nearest-neighbor duplicate
  audits are mandatory.

Risk: demo looks more clinically meaningful than the evidence allows.

- Check: safety language audit and failure-case view are part of done.

Risk: GB10 runtime is insufficient for full runs.

- Check: training budget has smoke, tiny, and full tiers; full MVP can reduce
  pair count or freeze more backbone layers, but cannot weaken baselines or
  leakage policy.

Go decision:

- full run artifact exists
- strongest baseline comparison is complete
- result wording matches measured evidence
- demo reads exported artifacts locally

No-go decision:

- leakage controls fail
- baselines are incomplete
- artifacts are not reproducible
- demo requires manual hidden data cleanup
- clinical language appears in result interpretation
