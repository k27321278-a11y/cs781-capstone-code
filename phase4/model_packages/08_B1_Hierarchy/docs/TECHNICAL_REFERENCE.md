# Technical reference — B1 document hierarchy

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

B1 6-encoder/6-decoder original Phase 1 backbone; 512 tokens per chunk, 16 ordered chunks per case, overlap 100, tail sampling. Two document Transformer blocks followed by bidirectional GRU. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

Data flow: Legal case → B1 6+6 backbone → 512×16 windows → Token evidence mix → Two document Transformer blocks followed by bidirectional GRU → Attention pool → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Staged B1 fine-tuning; document head mode `transformer_gru`.

### Provenance and scientific boundary

[CODE] B1: six-layer encoder and six-layer decoder from own Phase 1 weights; decoder representations supply token evidence.

[CODE] After token evidence pooling, a width-256 projection plus normalized position features enters two document Transformer blocks (four heads, width-768 gated feed-forward inner layer), then a 128-unit-per-direction bidirectional GRU. A learned document attention pools states into two logits.

[CODE/RUN] Staged fine-tuning starts at the task head and progressively unfreezes original encoder/decoder blocks. All optimizer parameter groups exist from the beginning; stage changes control which tensors receive gradients. Saved optimizer, scaler, RNG and position permit continuation between segments without reinitializing this head.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Backbone | `B1: six-layer encoder and six-layer decoder from own Phase 1 weights; decoder representations supply token evidence.` |
| Head mode | `transformer_gru` |
| Window policy | `tail` |
| Chunking | `16 × 512 tokens, overlap 100` |
| Document width | `256 for non-reference heads; B1 width for reference` |
| Learning rates | `backbone 1e-05; head 0.0003` |
| Accumulation | `8` |
| Warmup updates | `100` |
| Target epochs | `4` |
| Completed epochs | `2` |
| Optimizer steps | `8000` |
| Selected checkpoint | `0626c85191eb9e74a2133152e336b2b0cf9ad3e9a082b315eb2d129f3ac10a90` |

### Optimization and selection

[CODE] The B1 study uses cross-entropy for the binary label, AdamW with 0.01 weight decay, and staged unfreezing. Validation is performed on a fixed train-only holdout; checkpoint selection requires at least two epochs before patience can declare convergence. A time-budget stop instead records paused and a resumable checkpoint.

### Evaluation protocol

[RUN] The fixed 512-example train holdout selected model checkpoints at threshold 0.5. A separate, group-disjoint 512-example train calibration set selected the final decision threshold. The official 994-case dev evaluation used that threshold: macro-F1 0.534473; using 0.5 instead gave 0.499201. The split code excludes normalized-text duplicate groups across train, holdout, and calibration. The 1,517 test cases supplied no labels. A budget-paused run is not convergence.

### Reproduction lineage

B1 source and notebook versions are archived. The v7 source snapshot in code/source/ is the latest executable reference; for older v4 exports, the corresponding versioned notebook and run config identify the exact earlier run rather than implying that v7 code executed v4.

## Optimization, selection and evidence

Version 4 run state **complete**; 8000 optimizer updates and 2 complete epochs, selected step 6500. Calibrated official 994-case dev macro-F1 **0.534473** at threshold 0.486789; default 0.5 dev macro-F1 0.499201. Selected checkpoint SHA-256 `0626c85191eb9e74a2133152e336b2b0cf9ad3e9a082b315eb2d129f3ac10a90`; verified 1,517-ID ZIP SHA-256 `eb70fc73838a3fa84a6bd6ef5eac03ce680f62ef93bf780e8a9775ea7a384d7a`.

Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. No finished Codabench score for this exact export is established by the supplied screenshot.

## Limitations

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
