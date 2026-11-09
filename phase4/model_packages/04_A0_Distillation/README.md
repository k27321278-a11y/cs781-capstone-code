# A0 knowledge distillation

This is a **source-audited Phase 4 documentation package assembled after the training runs**. The illustrated notebook has no invented execution outputs. Its results are cited from preserved records.

![Architecture](figures/architecture.svg)

## Task and method

Binary Court Judgment Prediction from long legal case text. The twelve-layer causal A0 student learns CJPE labels and a frozen, larger, own D222 LoRMA teacher's class distribution. Teacher targets are created only for training IDs; temperature-scaled KL is mixed with supervised loss.

**Training/adaptation:** Full A0 student tuning with teacher soft targets

## Results and provenance

Original Kaggle run: complete; 700 optimizer updates; selected step 700. Official 994-case dev macro-F1 **0.563179** at recorded threshold 0.317929; default 0.5 dev macro-F1 0.367628. Verified prediction ZIP SHA-256 `4e26e303cdfc2681221be1e6fbf54be9fa07d114e4386f81cd89c128cf686ec2`, exactly 1,517 IDs. The teacher record and training-only target cache are copied in `evidence/`.

User-provided Codabench screenshot: **0.5691** (finished submission). Codabench is distinct from local official-dev macro-F1. The screenshot and provenance are indexed in the collection README.

## Files

- `code/`: exact source/config snapshot from the repository's Phase 4 code area; the preserved runner or builder is identified in `docs/CODE_MAP.md`.
- `notebooks/04_A0_Distillation_Explained.ipynb`: later illustrated, unexecuted companion. Original runnable notebooks, where available, are in `code/`.
- `evidence/`: compact copied original records with the selected configuration, training log and export verification.
- `submission/`: original verified prediction ZIP, not a report/code submission bundle.
- `report/`: methodology PDF, editable LaTeX body, and Markdown text.
- `figures/architecture.svg`: data flow.

## Reproducing

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`. Preserve the original data and checkpoint lineage, install PyTorch and the source's dependencies, map the Kaggle/local paths in the saved config, then run the original entry point. The compact GitHub package intentionally excludes large model weights and official course data. The original sources and downloaded checkpoints remain in the local Phase 4 archives.

## Interpretation

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage.
