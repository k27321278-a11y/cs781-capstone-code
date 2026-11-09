# B1 full-data reference

This is a **source-audited Phase 4 documentation package assembled after the training runs**. The illustrated notebook has no invented execution outputs. Its results are cited from preserved records.

![Architecture](figures/architecture.svg)

## Task and method

Binary Court Judgment Prediction from long legal case text. B1 6-encoder/6-decoder original Phase 1 backbone; 256 tokens per chunk, 8 ordered chunks per case, overlap 0, uniform sampling. Original evidence head: token evidence, ordered chunk BiGRU, attention/mean document mix. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

**Training/adaptation:** Staged B1 fine-tuning; document head mode `reference`

## Architecture and experiment details

B1: six-layer encoder and six-layer decoder from own Phase 1 weights; decoder representations supply token evidence. The reference reuses the shared evidence head: token mean/attention/last mixture, normalized position projection, bidirectional GRU over chunk vectors, attention/mean document mix, and two-class projection.

Staged fine-tuning starts at the task head and progressively unfreezes original encoder/decoder blocks. All optimizer parameter groups exist from the beginning; stage changes control which tensors receive gradients. Saved optimizer, scaler, RNG and position permit continuation between segments without reinitializing this head.

See [the technical reference](docs/TECHNICAL_REFERENCE.md) for the recorded configuration, validation boundary, and source-level mechanism. The [recorded trace](figures/recorded_trace.svg) is drawn from archived run records.

## Results and provenance

Version 4 run state **complete**; 14500 optimizer updates and 3 complete epochs, selected step 12500. Calibrated official 994-case dev macro-F1 **0.604727** at threshold 0.211745; default 0.5 dev macro-F1 0.634498. Selected checkpoint SHA-256 `5dadedc58ce5ecc89b8abe80b80c24a6165e2abb60108a81ea637bb3b3de1396`; verified 1,517-ID ZIP SHA-256 `fecc6fd3abe831f9156cf474dd5980a6edfc2c038eae4a84368d87c1b646be30`.

User-provided Codabench screenshot: **0.6145** (finished submission). Codabench is distinct from local official-dev macro-F1. The screenshot and provenance are indexed in the collection README.

## Files

- `code/`: exact source/config snapshot from the repository's Phase 4 code area; the preserved runner or builder is identified in `docs/CODE_MAP.md`.
- `notebooks/07_B1_Reference_Explained.ipynb`: later illustrated, unexecuted companion. Original runnable notebooks, where available, are in `code/`.
- `evidence/`: compact copied original records with the selected configuration, training log and export verification.
- `evidence/SOURCE_HASHES.json` and `evidence/README.md`: exact source hashes and an index of copied run records.
- `submission/`: original verified prediction ZIP, not a report/code submission bundle.
- `report/`: methodology PDF, editable LaTeX body, and Markdown text.
- `figures/architecture.svg`: data flow.

## Reproducing

Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`. Preserve the original data and checkpoint lineage, install PyTorch and the source's dependencies, map the Kaggle/local paths in the saved config, then run the original entry point. The compact GitHub package intentionally excludes large model weights and official course data. The original sources and downloaded checkpoints remain in the local Phase 4 archives.

## Interpretation

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub.
