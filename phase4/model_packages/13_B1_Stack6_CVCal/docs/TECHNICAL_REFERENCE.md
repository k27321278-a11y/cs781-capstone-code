# Technical reference — Six-model B1 stack with calibrated threshold

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism

This is a prediction-level ensemble, not a newly trained Phase 1/B1 backbone. It joins saved calibration, dev and test probabilities from: v7_transformer, v6_transformer, v5_gru, v4_reference, v7_gru, v6_gru. An L2-regularized logistic regression learns the mixing weights from 512 training-calibration examples; each original model retains its own checkpoint. Decision threshold 0.514611.

Data flow: B1 model probabilities → 6 inputs → 512 train calibration → L2 logistic weights → Threshold 0.515 → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Six B1 inputs; L2 logistic stack C=1.0 fit on 512 disjoint train calibration examples; decision threshold selected from five-fold out-of-fold calibration probabilities only. Official dev used for reporting; family was explored after other dev results, so cross-family comparison is exploratory. Hidden score unknown..

## Optimization, selection and evidence

Five-fold train-calibration CV macro-F1 **0.716744** on 512 cases. Exploratory official 994-case dev macro-F1 **0.701977**. Verified 1,517-ID ZIP SHA-256 `e879cdf4f9bc3bef32baa398e4fd931a765589297f7899a2987916f663538ef4`.

Run the original builder `code/build_b1_stack6_cvcal_submission.py` with the exact model probability arrays named and hashed in `evidence/VERIFIED_B1_STACK6_CVCAL.json`. The record also stores learned coefficients/intercept.

## Limitations

The stack was explored after inspecting other dev results, so comparisons across stack families on the same official dev set are post-selection and may be optimistic. The train-calibration CV is not an unseen test estimate. Reproduction needs the saved per-model probability arrays and selected checkpoint lineage in the original local B1 archive; the small package includes the builder script and full hash/provenance record, but not all probabilities or model weights. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
