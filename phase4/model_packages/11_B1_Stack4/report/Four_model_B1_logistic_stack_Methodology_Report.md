# Four-model B1 logistic stack — Phase 4 CJPE methodology

## Problem statement and data

Phase 4 Court Judgment Prediction predicts binary outcome 0/1 from legal case text. The preserved official inputs are Train/train.json, Train/dev.json and Test/test.json. Test labels are private. Original run records report 994 official dev labels and 1,517 test IDs; macro-F1 is the central metric.

## Architecture and mechanics

This is a prediction-level ensemble, not a newly trained Phase 1/B1 backbone. It joins saved calibration, dev and test probabilities from: v7_transformer, v6_transformer, v5_gru, v4_reference. An L2-regularized logistic regression learns the mixing weights from 512 training-calibration examples; each original model retains its own checkpoint. Decision threshold 0.500000. Data flow: B1 model probabilities -> 4 inputs -> 512 train calibration -> L2 logistic weights -> Threshold 0.500 -> Binary ZIP.

## Training design and checkpoint selection

Four inputs fixed from prior evidence; L2 logistic stack C=0.1 fit on 512 disjoint train calibration examples; five-fold calibration CV; threshold 0.5. Official dev used only to report.. Run the original builder `code/build_b1_stack_submission.py` with the exact model probability arrays named and hashed in `evidence/VERIFIED_B1_STACK_V7V6V5V4.json`. The record also stores learned coefficients/intercept.

## Stack mechanics and coefficient provenance

Input model names in order: v7_transformer, v6_transformer, v5_gru, v4_reference. Each input contributes a probability column for aligned case IDs. An L2-penalized logistic regression fits 512 train-calibration examples; the record stores coefficients [[0.556751645610894, 0.3330229606145827, 0.3001991480071267, 0.3512995187063114]] and intercept [-0.03396466502718611]. For each case, the fitted linear log-odds combine base probabilities, the logistic sigmoid maps the result to a score, and a fixed or calibration-selected threshold converts it to label 0/1. The score threshold is 0.500000; its selection rule and exact hashes for all input arrays are in the copied verification record.

## Measured results

Five-fold train-calibration CV macro-F1 **0.710655** on 512 cases. Exploratory official 994-case dev macro-F1 **0.693956**. Verified 1,517-ID ZIP SHA-256 `b341bb9d03656077c32ffd77cf4c60f9cb38ac55f6d7e583cb40080d0761e85e`. User-provided Codabench screenshot: **0.6723** (finished submission).

## Validation and selection

Five-fold cross-validation over the 512 train-calibration cases yielded macro-F1 0.710655. Official-dev macro-F1 0.693956 on 994 cases is a separate report-only observation. Comparing several stack variants on the same dev data introduces post-selection optimism. The 1,517 test predictions were made without test labels.

## Prediction contract and verification

The original prediction ZIP contains one root-level prediction.json with 1,517 unique official test IDs and integer labels restricted to 0 and 1. This package checks the ZIP CRC and SHA-256 against the archived run/verification record. Full model checkpoint tensors remain outside GitHub; their downloaded backup receipts and hashes live in the local Phase 4 study archives.

## Reproducibility and limits

The stack was explored after inspecting other dev results, so comparisons across stack families on the same official dev set are post-selection and may be optimistic. The train-calibration CV is not an unseen test estimate. Reproduction needs the saved per-model probability arrays and selected checkpoint lineage in the original local B1 archive; the small package includes the builder script and full hash/provenance record, but not all probabilities or model weights. Reports and companion notebook were created after training; original scores come from copied artifacts. Model weights and course data are preserved locally, not uploaded to GitHub.
