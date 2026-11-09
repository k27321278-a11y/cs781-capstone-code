# Technical reference — C1 BitFit

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

Only bias parameters in the C1 backbone receive gradients, with a trainable task evidence head. This tests an inexpensive adaptation of the same recurrent hybrid used by C1 full tuning.

Data flow: Legal case → C1 own tokenizer/backbone → Bias-only backbone tuning → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Bias-only backbone tuning.

## Optimization, selection and evidence

Original Kaggle run: complete; 800 optimizer updates; selected step 400. Official 994-case dev macro-F1 **0.535560** at recorded threshold 0.449234; default 0.5 dev macro-F1 0.454764. Verified prediction ZIP SHA-256 `6d5be1e9fab3c623b9bb0323c54a5a95ab01f6421507a8fd9712e37c135d7c15`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
