# Technical reference — D222 + LoRMA

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

The 24-layer D222 causal decoder supplies contextual token states. Multiplicative low-rank updates adapt selected weights, while the evidence head learns token salience and ordered case-level context.

Data flow: Legal case → D222 own tokenizer/backbone → Rank-8 multiplicative low-rank adaptation → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Rank-8 multiplicative low-rank adaptation.

## Optimization, selection and evidence

Original Kaggle run: complete; 213 optimizer updates; selected step 213. Official 994-case dev macro-F1 **0.449472** at recorded threshold 0.284564; default 0.5 dev macro-F1 0.333333. Verified prediction ZIP SHA-256 `3be58f83f018fb99b1e491e0b69ff4507581b005de2f19d5fc52b8d01aa5af28`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
