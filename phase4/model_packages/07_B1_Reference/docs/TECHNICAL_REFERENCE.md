# Technical reference — B1 full-data reference

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

B1 6-encoder/6-decoder original Phase 1 backbone; 256 tokens per chunk, 8 ordered chunks per case, overlap 0, uniform sampling. Original evidence head: token evidence, ordered chunk BiGRU, attention/mean document mix. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

Data flow: Legal case → B1 6+6 backbone → 256×8 windows → Token evidence mix → Original evidence head: token evidence, ordered chunk BiGRU, attention/mean document mix → Attention pool → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Staged B1 fine-tuning; document head mode `reference`. Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`..

## Optimization, selection and evidence

Version 4 run state **complete**; 14500 optimizer updates and 3 complete epochs, selected step 12500. Calibrated official 994-case dev macro-F1 **0.604727** at threshold 0.211745; default 0.5 dev macro-F1 0.634498. Selected checkpoint SHA-256 `5dadedc58ce5ecc89b8abe80b80c24a6165e2abb60108a81ea637bb3b3de1396`; verified 1,517-ID ZIP SHA-256 `fecc6fd3abe831f9156cf474dd5980a6edfc2c038eae4a84368d87c1b646be30`.

Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`.

## Limitations

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
