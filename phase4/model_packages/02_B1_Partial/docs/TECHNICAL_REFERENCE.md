# Technical reference — B1 partial fine-tuning

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

The original six-encoder/six-decoder B1 backbone is partially unfrozen. The shared evidence head pools tokens and models ordered chunks with a bidirectional GRU.

Data flow: Legal case → B1 own tokenizer/backbone → Last two encoder and decoder blocks, plus norms → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Last two encoder and decoder blocks, plus norms.

## Optimization, selection and evidence

Original Kaggle run: complete; 1233 optimizer updates; selected step 1100. Official 994-case dev macro-F1 **0.594530** at recorded threshold 0.730290; default 0.5 dev macro-F1 0.397437. Verified prediction ZIP SHA-256 `d9acd7b64973e6e815f743003eca864ff7b569590bdf53fcef743f1cc996728a`, exactly 1,517 IDs.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
