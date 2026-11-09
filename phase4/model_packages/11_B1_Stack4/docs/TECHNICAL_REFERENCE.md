# Technical reference — Four-model B1 logistic stack

## Problem and measured splits

Phase 4 predicts a binary outcome (0 or 1) from a legal case. The provided files are `Train/train.json`, `Train/dev.json`, and `Test/test.json`. The original Kaggle run records report 994 official labeled dev examples and 1,517 test IDs. Test labels are withheld. The primary reported metric is macro-F1.

## Mechanism [CODE]

This is a prediction-level ensemble, not a newly trained Phase 1/B1 backbone. It joins saved calibration, dev and test probabilities from: v7_transformer, v6_transformer, v5_gru, v4_reference. An L2-regularized logistic regression learns the mixing weights from 512 training-calibration examples; each original model retains its own checkpoint. Decision threshold 0.500000.

Data flow: B1 model probabilities → 4 inputs → 512 train calibration → L2 logistic weights → Threshold 0.500 → Binary ZIP. The source under `code/` is the authority for exact tensor operations and dimensions. Four inputs fixed from prior evidence; L2 logistic stack C=0.1 fit on 512 disjoint train calibration examples; five-fold calibration CV; threshold 0.5. Official dev used only to report..

### Provenance and scientific boundary

[CODE] Prediction-level B1 ensemble, not an additional neural backbone.

[CODE] For a case, stack the base-model probabilities into vector p. Compute logit = intercept + Σ(coefficient_i × p_i), apply sigmoid, and output 1 if the resulting probability reaches the recorded threshold. Every base probability comes from an independently trained B1 model; no backbone weight is updated when fitting the stack.

[CODE/RUN] L2-regularized logistic regression combines 4 aligned model probabilities: v7_transformer, v6_transformer, v5_gru, v4_reference. The stored coefficients are 0.5568, 0.3330, 0.3002, 0.3513 and intercept -0.0340.

### Exact recorded configuration

| Setting | Recorded value |
|---|---|
| Input models | `v7_transformer, v6_transformer, v5_gru, v4_reference` |
| Calibration rows | `512` |
| Regularizer | `L2 logistic regression` |
| Decision threshold | `0.5` |
| 5-fold calibration CV macro-F1 | `0.7106549364613881` |
| Official dev macro-F1 | `0.6939555987684329` |
| Test predictions | `1517` |
| Output ZIP SHA-256 | `b341bb9d03656077c32ffd77cf4c60f9cb38ac55f6d7e583cb40080d0761e85e` |

### Optimization and selection

[CODE] There are no optimizer epochs for a new backbone here. The logistic combiner fits probability columns with L2 regularization. The stack with CV-calibrated threshold chooses its cutoff using out-of-fold calibration scores; the other stacks use 0.5.

### Evaluation protocol

[RUN] The stack was fitted only on 512 train-calibration rows, and five-fold calibration CV macro-F1 was 0.710655. Official dev macro-F1 0.693956 on 994 cases is exploratory because multiple stack families were compared after seeing dev. The hidden Codabench score is a separate user-supplied observation; calibration CV is not an unseen test result.

### Reproduction lineage

The original builder script and verified input-array hashes are in code/ and evidence/. Original base-model probability arrays and checkpoint backups remain in the local B1 study archive.

## Optimization, selection and evidence

Five-fold train-calibration CV macro-F1 **0.710655** on 512 cases. Exploratory official 994-case dev macro-F1 **0.693956**. Verified 1,517-ID ZIP SHA-256 `b341bb9d03656077c32ffd77cf4c60f9cb38ac55f6d7e583cb40080d0761e85e`.

Run the original builder `code/build_b1_stack_submission.py` with the exact model probability arrays named and hashed in `evidence/VERIFIED_B1_STACK_V7V6V5V4.json`. The record also stores learned coefficients/intercept.

## Evidence classes

`[CODE]` denotes source-inspected behavior; `[RUN]` denotes preserved configuration or metrics; `[SCREENSHOT]` denotes the user-provided Codabench submission table; `[INTERPRETATION]` denotes an explanatory inference. User-provided Codabench screenshot: **0.6723** (finished submission).

## Limitations

The stack was explored after inspecting other dev results, so comparisons across stack families on the same official dev set are post-selection and may be optimistic. The train-calibration CV is not an unseen test estimate. Reproduction needs the saved per-model probability arrays and selected checkpoint lineage in the original local B1 archive; the small package includes the builder script and full hash/provenance record, but not all probabilities or model weights. This package contains a later explanatory notebook, not a notebook with original saved execution outputs. GitHub excludes checkpoint tensors and official datasets; their original local backups are preserved. Do not infer a hidden leaderboard score from official-dev performance.
