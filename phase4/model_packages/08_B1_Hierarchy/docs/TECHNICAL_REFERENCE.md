# Technical reference — B1 document hierarchy

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

B1 6-encoder/6-decoder original Phase 1 backbone; 512 tokens per chunk, 16 ordered chunks per case, overlap 100, tail sampling. Two document Transformer blocks followed by bidirectional GRU. Token representation uses a learned mixture of mean, attention and last content token; position enters before document aggregation. The final learned attention pools document states for two-class logits.

Data flow: Legal case → B1 6+6 backbone → 512×16 windows → Token evidence mix → Two document Transformer blocks followed by bidirectional GRU → Attention pool → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Staged B1 fine-tuning; document head mode `transformer_gru`. Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`..

## Optimization, selection and evidence

Version 4 run state **complete**; 8000 optimizer updates and 2 complete epochs, selected step 6500. Calibrated official 994-case dev macro-F1 **0.534473** at threshold 0.486789; default 0.5 dev macro-F1 0.499201. Selected checkpoint SHA-256 `0626c85191eb9e74a2133152e336b2b0cf9ad3e9a082b315eb2d129f3ac10a90`; verified 1,517-ID ZIP SHA-256 `eb70fc73838a3fa84a6bd6ef5eac03ce680f62ef93bf780e8a9775ea7a384d7a`.

Original B1 study notebook lineage `code/v1.ipynb` through `code/v4.ipynb`; latest source snapshot `code/source/b1_study/`. Saved configs and terminal/run evidence are in `evidence/`.

## Limitations

A budget-paused run is not converged; the target was four epochs with checkpoint selection and train-only calibration. The v7 source also defines gated depth expansion and an upper FFN option, but the recorded configs for these exports keep encoder_depth=6 and expand_ffn=false; those unlaunched options have no reported score. Original checkpoint bytes were downloaded and verified in the local archive, but are omitted from GitHub. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
