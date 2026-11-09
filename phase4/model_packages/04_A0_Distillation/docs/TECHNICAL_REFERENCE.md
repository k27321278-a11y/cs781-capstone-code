# Technical reference — A0 knowledge distillation

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

The twelve-layer causal A0 student learns CJPE labels and a frozen, larger, own D222 LoRMA teacher's class distribution. Teacher targets are created only for training IDs; temperature-scaled KL is mixed with supervised loss.

Data flow: Legal case → A0 own tokenizer/backbone → Full A0 student tuning with teacher soft targets → Token evidence pooling → Ordered chunk BiGRU → 2-class logits → prediction.json. The source under `code/` is the authority for exact tensor operations and dimensions. Full A0 student tuning with teacher soft targets.

## Optimization, selection and evidence

Original Kaggle run: complete; 700 optimizer updates; selected step 700. Official 994-case dev macro-F1 **0.563179** at recorded threshold 0.317929; default 0.5 dev macro-F1 0.367628. Verified prediction ZIP SHA-256 `4e26e303cdfc2681221be1e6fbf54be9fa07d114e4386f81cd89c128cf686ec2`, exactly 1,517 IDs. The teacher record and training-only target cache are copied in `evidence/`.

Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Limitations

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
