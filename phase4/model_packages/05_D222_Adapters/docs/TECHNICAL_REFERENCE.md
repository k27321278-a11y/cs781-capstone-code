# Technical reference — D222 bottleneck adapters

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

A zero-initialized up projection lets each width-32 bottleneck begin as an identity residual on selected feed-forward output layers. The task evidence head is trained alongside the adapters.

Data flow: Legal case → D222 own tokenizer/backbone → Frozen D222 with width-32 residual FFN adapters → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Frozen D222 with width-32 residual FFN adapters.

### Provenance and scientific boundary

[CODE] D222: 24-layer causal decoder with grouped-query attention and alternating local/global context

[CODE] For each unpadded chunk, the original backbone yields contextual token states. The head layer-normalizes them, then makes a softmax-weighted mixture of token mean, learned token attention and the last token. Normalized chunk positions enter through a learned projection. A bidirectional GRU contextualizes chunks in document order; learned attention and a chunk mean are mixed, then LayerNorm/dropout and a two-logit classifier predict the binary judgment.

[CODE/RUN] Every selected feed-forward w2 output gets a width-32 down/GELU/up residual adapter. Zero initialization of the up projection makes the initial addition zero while the original backbone is frozen.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Family | `D222: 24-layer causal decoder with grouped-query attention and alternating local/global context` |
| Adaptation | `adapters` |
| Train case cap | `12000` |
| Train-only holdout | `256 per class` |
| Chunking | `8 × 256 tokens; 8192 source characters` |
| Learning rates | `backbone 0.0003; head 0.0003` |
| Accumulation | `8` |
| Optimizer steps | `353` |
| Parameters | `224,620,682 total; 2,540,810 trainable` |
| Selected checkpoint | `d08fd295c77a6e4231fedae2f8dac13258e285434bff31cd7a77209018f12840` |
| Data fingerprint | `88913c10c288a6f77a030d0391904d40b645fc242f2774f0ef8216e9f5afa792` |

### Optimization and selection

[CODE] The runner uses AdamW with separate backbone/head learning rates and weight decay 0.01. For each training example it computes class-weighted cross-entropy; weights are capped at 3.0. It periodically evaluates the fixed train-only holdout, saves the best checkpoint, and exports predictions from selected weights.

### Evaluation protocol

[RUN] A fixed, balanced training-only holdout of up to 512 cases selected the checkpoint and decision threshold. The official 994-case labeled dev set was evaluated after that selection; selected-threshold dev macro-F1 was 0.530606, while default 0.5 gave 0.442347. The hidden test labels were not available. These dev metrics are not the Codabench score shown in the user screenshot.

### Reproduction lineage

Initial six-job Kaggle v3 runner; the per-model notebook in code/ is prepared source, while v3_t4x2.ipynb and archived Kaggle outputs identify the executed run.

## Optimization, selection and evidence

Original Kaggle run: complete; 353 optimizer updates; selected step 353. Official 994-case dev macro-F1 **0.530606** at recorded threshold 0.367270; default 0.5 dev macro-F1 0.442347. Verified prediction ZIP SHA-256 `3c0c236761a6fdc6d87cfac3ebf94cbc518e3f5229b89a02c50f7e888250c35d`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. No finished Codabench score for this exact export is established by the supplied screenshot.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
