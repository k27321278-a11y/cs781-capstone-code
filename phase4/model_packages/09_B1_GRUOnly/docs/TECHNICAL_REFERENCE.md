# Technical reference — B1 window GRU-only

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

B1 6-encoder/6-decoder original Phase 1 backbone; 512 tokens per chunk, 16 ordered chunks per case, overlap 100, tail sampling. Document BiGRU without document Transformer. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

Data flow: Legal case → B1 6+6 backbone → 512×16 windows → Token evidence mix → Document BiGRU without document Transformer → Attention pool → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Staged B1 fine-tuning; document head mode `gru_only`. Original B1 study notebook lineage `code/v1.ipynb` through `code/v7.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`..

## Optimization, selection and evidence

Version 7 run state **paused**; 7465 optimizer updates and 1 complete epochs, selected step 5000. Calibrated official 994-case dev macro-F1 **0.619677** at threshold 0.447596; default 0.5 dev macro-F1 0.649809. Selected checkpoint SHA-256 `d01f8978e9ac2d13eee5d202fa75df285c27915fecdbbc0909f72703466d8c80`; verified 1,517-ID ZIP SHA-256 `796c249f600dbd3140102ce45fd797b397ccc2277bcf1b11693167fc9f2a5455`.

Original B1 study notebook lineage `code/v1.ipynb` through `code/v7.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`.

## Limitations

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
