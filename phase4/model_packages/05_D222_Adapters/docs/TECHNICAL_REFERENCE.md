# Technical reference — D222 bottleneck adapters

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

A zero-initialized up projection lets each width-32 bottleneck begin as an identity residual on selected feed-forward output layers. The task evidence head is trained alongside the adapters.

Data flow: Legal case → D222 own tokenizer/backbone → Frozen D222 with width-32 residual FFN adapters → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Frozen D222 with width-32 residual FFN adapters.

## Optimization, selection and evidence

Original Kaggle run: complete; 353 optimizer updates; selected step 353. Official 994-case dev macro-F1 **0.530606** at recorded threshold 0.367270; default 0.5 dev macro-F1 0.442347. Verified prediction ZIP SHA-256 `3c0c236761a6fdc6d87cfac3ebf94cbc518e3f5229b89a02c50f7e888250c35d`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
