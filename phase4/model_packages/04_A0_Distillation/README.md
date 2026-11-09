# A0 knowledge distillation

This is a **source-audited Phase 4 documentation package assembled after the training runs**. The illustrated notebook has no invented execution outputs. Its results are cited from preserved records.

![Architecture](figures/architecture.svg)

## Task and method

Binary Court Judgment Prediction from long legal case text. The twelve-layer causal A0 student learns CJPE labels and a frozen, larger, own D222 LoRMA teacher's class distribution. Teacher targets are created only for training IDs; temperature-scaled KL is mixed with supervised loss.

**Training/adaptation:** Full A0 student tuning with teacher soft targets

## Architecture and experiment details

A0: 12-layer causal Transformer decoder For each unpadded chunk, the original backbone yields contextual token states. The head layer-normalizes them, then makes a softmax-weighted mixture of token mean, learned token attention and the last token. Normalized chunk positions enter through a learned projection. A bidirectional GRU contextualizes chunks in document order; learned attention and a chunk mean are mixed, then LayerNorm/dropout and a two-logit classifier predict the binary judgment.

All original backbone weights are trainable, along with the task head. A0 is a full-tuned student. Frozen D222 LoRMA teacher logits are cached on training IDs only. The student loss mixes supervised cross-entropy and temperature-2 KL, each at weight 0.5; teacher targets do not include dev/test labels.

See [the technical reference](docs/TECHNICAL_REFERENCE.md) for the recorded configuration, validation boundary, and source-level mechanism. The [recorded trace](figures/recorded_trace.svg) is drawn from archived run records.

## Results and provenance

Original Kaggle run: complete; 700 optimizer updates; selected step 700. Official 994-case dev macro-F1 **0.563179** at recorded threshold 0.317929; default 0.5 dev macro-F1 0.367628. Verified prediction ZIP SHA-256 `4e26e303cdfc2681221be1e6fbf54be9fa07d114e4386f81cd89c128cf686ec2`, exactly 1,517 IDs. The teacher record and training-only target cache are copied in `evidence/`.

User-provided Codabench screenshot: **0.5691** (finished submission). Codabench is distinct from local official-dev macro-F1. The screenshot and provenance are indexed in the collection README.

## Files

- `code/`: exact source/config snapshot from the repository's Phase 4 code area; the preserved runner or builder is identified in `docs/CODE_MAP.md`.
- `notebooks/04_A0_Distillation_Explained.ipynb`: later illustrated, unexecuted companion. Original runnable notebooks, where available, are in `code/`.
- `evidence/`: compact copied original records with the selected configuration, training log and export verification.
- `evidence/SOURCE_HASHES.json` and `evidence/README.md`: exact source hashes and an index of copied run records.
- `submission/`: original verified prediction ZIP, not a report/code submission bundle.
- `report/`: methodology PDF, editable LaTeX body, and Markdown text.
- `figures/architecture.svg`: data flow.

## Reproducing

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`. Preserve the original data and checkpoint lineage, install PyTorch and the source's dependencies, map the Kaggle/local paths in the saved config, then run the original entry point. The compact GitHub package intentionally excludes large model weights and official course data. The original sources and downloaded checkpoints remain in the local Phase 4 archives.

## Interpretation

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage.
