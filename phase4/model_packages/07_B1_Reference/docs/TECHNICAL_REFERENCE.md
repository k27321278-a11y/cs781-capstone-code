# Technical reference — B1 full-data reference

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

B1 6-encoder/6-decoder original Phase 1 backbone; 256 tokens per chunk, 8 ordered chunks per case, overlap 0, uniform sampling. Original evidence head: token evidence, ordered chunk BiGRU, attention/mean document mix. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

Data flow: Legal case → B1 6+6 backbone → 256×8 windows → Token evidence mix → Original evidence head: token evidence, ordered chunk BiGRU, attention/mean document mix → Attention pool → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Staged B1 fine-tuning; document head mode `reference`.

### Provenance and scientific boundary

[CODE] B1: six-layer encoder and six-layer decoder from own Phase 1 weights; decoder representations supply token evidence.

[CODE] The reference reuses the shared evidence head: token mean/attention/last mixture, normalized position projection, bidirectional GRU over chunk vectors, attention/mean document mix, and two-class projection.

[CODE/RUN] Staged fine-tuning starts at the task head and progressively unfreezes original encoder/decoder blocks. All optimizer parameter groups exist from the beginning; stage changes control which tensors receive gradients. Saved optimizer, scaler, RNG and position permit continuation between segments without reinitializing this head.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Backbone | `B1: six-layer encoder and six-layer decoder from own Phase 1 weights; decoder representations supply token evidence.` |
| Head mode | `reference` |
| Window policy | `uniform` |
| Chunking | `8 × 256 tokens, overlap 0` |
| Document width | `256 for non-reference heads; B1 width for reference` |
| Learning rates | `backbone 1e-05; head 0.0003` |
| Accumulation | `8` |
| Warmup updates | `100` |
| Target epochs | `4` |
| Completed epochs | `3` |
| Optimizer steps | `14500` |
| Selected checkpoint | `5dadedc58ce5ecc89b8abe80b80c24a6165e2abb60108a81ea637bb3b3de1396` |

### Optimization and selection

[CODE] The B1 study uses cross-entropy for the binary label, AdamW with 0.01 weight decay, and staged unfreezing. Validation is performed on a fixed train-only holdout; checkpoint selection requires at least two epochs before patience can declare convergence. A time-budget stop instead records paused and a resumable checkpoint.

### Evaluation protocol

[RUN] The fixed 512-example train holdout selected model checkpoints at threshold 0.5. A separate, group-disjoint 512-example train calibration set selected the final decision threshold. The official 994-case dev evaluation used that threshold: macro-F1 0.604727; using 0.5 instead gave 0.634498. The split code excludes normalized-text duplicate groups across train, holdout, and calibration. The 1,517 test cases supplied no labels. A budget-paused run is not convergence.

### Reproduction lineage

B1 source and notebook versions are archived. The v7 source snapshot in code/source/ is the latest executable reference; for older v4 exports, the corresponding versioned notebook and run config identify the exact earlier run rather than implying that v7 code executed v4.

## Optimization, selection and evidence

Version 4 run state **complete**; 14500 optimizer updates and 3 complete epochs, selected step 12500. Calibrated official 994-case dev macro-F1 **0.604727** at threshold 0.211745; default 0.5 dev macro-F1 0.634498. Selected checkpoint SHA-256 `5dadedc58ce5ecc89b8abe80b80c24a6165e2abb60108a81ea637bb3b3de1396`; verified 1,517-ID ZIP SHA-256 `fecc6fd3abe831f9156cf474dd5980a6edfc2c038eae4a84368d87c1b646be30`.

Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. User-provided Codabench screenshot: **0.6145** (finished submission).

## Limitations

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
