# Technical reference — B1 partial fine-tuning

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

The original six-encoder/six-decoder B1 backbone is partially unfrozen. The shared evidence head pools tokens and models ordered chunks with a bidirectional GRU.

Data flow: Legal case → B1 own tokenizer/backbone → Last two encoder and decoder blocks, plus norms → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Last two encoder and decoder blocks, plus norms.

### Provenance and scientific boundary

[CODE] B1: six-layer encoder and six-layer decoder with cross-attention

[CODE] For each unpadded chunk, the original backbone yields contextual token states. The head layer-normalizes them, then makes a softmax-weighted mixture of token mean, learned token attention and the last token. Normalized chunk positions enter through a learned projection. A bidirectional GRU contextualizes chunks in document order; learned attention and a chunk mean are mixed, then LayerNorm/dropout and a two-logit classifier predict the binary judgment.

[CODE/RUN] Only the last two encoder blocks, last two decoder blocks, and encoder/decoder final norms of B1 are unfrozen; the other backbone blocks are frozen.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Family | `B1: six-layer encoder and six-layer decoder with cross-attention` |
| Adaptation | `partial` |
| Train case cap | `12000` |
| Train-only holdout | `256 per class` |
| Chunking | `8 × 256 tokens; 8192 source characters` |
| Learning rates | `backbone 2e-05; head 0.0003` |
| Accumulation | `8` |
| Optimizer steps | `1233` |
| Parameters | `125,274,250 total; 34,032,778 trainable` |
| Selected checkpoint | `4c4ca537ec9a02bac12b577cf73b5fb7db29fff9eb175b15103a30bd291f1977` |
| Data fingerprint | `88913c10c288a6f77a030d0391904d40b645fc242f2774f0ef8216e9f5afa792` |

### Optimization and selection

[CODE] The runner uses AdamW with separate backbone/head learning rates and weight decay 0.01. For each training example it computes class-weighted cross-entropy; weights are capped at 3.0. It periodically evaluates the fixed train-only holdout, saves the best checkpoint, and exports predictions from selected weights.

### Evaluation protocol

[RUN] A fixed, balanced training-only holdout of up to 512 cases selected the checkpoint and decision threshold. The official 994-case labeled dev set was evaluated after that selection; selected-threshold dev macro-F1 was 0.594530, while default 0.5 gave 0.397437. The hidden test labels were not available. These dev metrics are not the Codabench score shown in the user screenshot.

### Reproduction lineage

Initial six-job Kaggle v3 runner; the per-model notebook in code/ is prepared source, while v3_t4x2.ipynb and archived Kaggle outputs identify the executed run.

## Optimization, selection and evidence

Original Kaggle run: complete; 1233 optimizer updates; selected step 1100. Official 994-case dev macro-F1 **0.594530** at recorded threshold 0.730290; default 0.5 dev macro-F1 0.397437. Verified prediction ZIP SHA-256 `d9acd7b64973e6e815f743003eca864ff7b569590bdf53fcef743f1cc996728a`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. User-provided Codabench screenshot: **0.5895** (finished submission).

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
