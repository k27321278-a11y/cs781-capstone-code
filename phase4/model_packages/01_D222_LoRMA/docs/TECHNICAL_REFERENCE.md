# Technical reference — D222 + LoRMA

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

The 24-layer D222 causal decoder supplies contextual token states. Multiplicative low-rank updates adapt selected weights, while the evidence head learns token salience and ordered case-level context.

Data flow: Legal case → D222 own tokenizer/backbone → Rank-8 multiplicative low-rank adaptation → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Rank-8 multiplicative low-rank adaptation.

### Provenance and scientific boundary

[CODE] D222: 24-layer causal decoder with grouped-query attention and alternating local/global context

[CODE] For each unpadded chunk, the original backbone yields contextual token states. The head layer-normalizes them, then makes a softmax-weighted mixture of token mean, learned token attention and the last token. Normalized chunk positions enter through a learned projection. A bidirectional GRU contextualizes chunks in document order; learned attention and a chunk mean are mixed, then LayerNorm/dropout and a two-logit classifier predict the binary judgment.

[CODE/RUN] LoRMA multiplies selected original linear transforms by trained low-rank factors (rank 8, alpha 16 here); the unchanged parent checkpoint remains the initialization.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Family | `D222: 24-layer causal decoder with grouped-query attention and alternating local/global context` |
| Adaptation | `lorma` |
| Train case cap | `12000` |
| Train-only holdout | `256 per class` |
| Chunking | `8 × 256 tokens; 8192 source characters` |
| Learning rates | `backbone 0.0002; head 0.0003` |
| Accumulation | `8` |
| Optimizer steps | `213` |
| Parameters | `226,171,274 total; 4,091,402 trainable` |
| Selected checkpoint | `002e0dedacc5e8283ad8519c2ee6e6c9e3af15c83e3bf4cc2ee61178499b151a` |
| Data fingerprint | `88913c10c288a6f77a030d0391904d40b645fc242f2774f0ef8216e9f5afa792` |

### Optimization and selection

[CODE] The runner uses AdamW with separate backbone/head learning rates and weight decay 0.01. For each training example it computes class-weighted cross-entropy; weights are capped at 3.0. It periodically evaluates the fixed train-only holdout, saves the best checkpoint, and exports predictions from selected weights.

### Evaluation protocol

[RUN] A fixed, balanced training-only holdout of up to 512 cases selected the checkpoint and decision threshold. The official 994-case labeled dev set was evaluated after that selection; selected-threshold dev macro-F1 was 0.449472, while default 0.5 gave 0.333333. The hidden test labels were not available. These dev metrics are not the Codabench score shown in the user screenshot.

### Reproduction lineage

Initial six-job Kaggle v3 runner; the per-model notebook in code/ is prepared source, while v3_t4x2.ipynb and archived Kaggle outputs identify the executed run.

## Optimization, selection and evidence

Original Kaggle run: complete; 213 optimizer updates; selected step 213. Official 994-case dev macro-F1 **0.449472** at recorded threshold 0.284564; default 0.5 dev macro-F1 0.333333. Verified prediction ZIP SHA-256 `3be58f83f018fb99b1e491e0b69ff4507581b005de2f19d5fc52b8d01aa5af28`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. User-provided Codabench screenshot: **0.4650** (finished submission).

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
