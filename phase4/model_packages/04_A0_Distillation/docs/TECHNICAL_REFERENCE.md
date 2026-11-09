# Technical reference — A0 knowledge distillation

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

The twelve-layer causal A0 student learns CJPE labels and a frozen, larger, own D222 LoRMA teacher's class distribution. Teacher targets are created only for training IDs; temperature-scaled KL is mixed with supervised loss.

Data flow: Legal case → A0 own tokenizer/backbone → Full A0 student tuning with teacher soft targets → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Full A0 student tuning with teacher soft targets.

### Provenance and scientific boundary

[CODE] A0: 12-layer causal Transformer decoder

[CODE] For each unpadded chunk, the original backbone yields contextual token states. The head layer-normalizes them, then makes a softmax-weighted mixture of token mean, learned token attention and the last token. Normalized chunk positions enter through a learned projection. A bidirectional GRU contextualizes chunks in document order; learned attention and a chunk mean are mixed, then LayerNorm/dropout and a two-logit classifier predict the binary judgment.

[CODE/RUN] All original backbone weights are trainable, along with the task head. A0 is a full-tuned student. Frozen D222 LoRMA teacher logits are cached on training IDs only. The student loss mixes supervised cross-entropy and temperature-2 KL, each at weight 0.5; teacher targets do not include dev/test labels.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Family | `A0: 12-layer causal Transformer decoder` |
| Adaptation | `full` |
| Train case cap | `12000` |
| Train-only holdout | `256 per class` |
| Chunking | `8 × 256 tokens; 8192 source characters` |
| Learning rates | `backbone 1e-05; head 0.0003` |
| Accumulation | `8` |
| Optimizer steps | `700` |
| Parameters | `111,113,098 total; 111,113,098 trainable` |
| Selected checkpoint | `47df27c673d6ee7619853765d7ad3112ece96bfc9bee9780f30dc606bce3de98` |
| Data fingerprint | `88913c10c288a6f77a030d0391904d40b645fc242f2774f0ef8216e9f5afa792` |

### Optimization and selection

[CODE] The runner uses AdamW with separate backbone/head learning rates and weight decay 0.01. For each training example it computes class-weighted cross-entropy; weights are capped at 3.0. It periodically evaluates the fixed train-only holdout, saves the best checkpoint, and exports predictions from selected weights. The A0 student additionally uses the teacher KL term described above.

### Evaluation protocol

[RUN] A fixed, balanced training-only holdout of up to 512 cases selected the checkpoint and decision threshold. The official 994-case labeled dev set was evaluated after that selection; selected-threshold dev macro-F1 was 0.563179, while default 0.5 gave 0.367628. The hidden test labels were not available. These dev metrics are not the Codabench score shown in the user screenshot.

### Reproduction lineage

Initial six-job Kaggle v3 runner; the per-model notebook in code/ is prepared source, while v3_t4x2.ipynb and archived Kaggle outputs identify the executed run.

## Optimization, selection and evidence

Original Kaggle run: complete; 700 optimizer updates; selected step 700. Official 994-case dev macro-F1 **0.563179** at recorded threshold 0.317929; default 0.5 dev macro-F1 0.367628. Verified prediction ZIP SHA-256 `4e26e303cdfc2681221be1e6fbf54be9fa07d114e4386f81cd89c128cf686ec2`, exactly 1,517 IDs. The teacher record and training-only target cache are copied in `evidence/`.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. User-provided Codabench screenshot: **0.5691** (finished submission).

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
