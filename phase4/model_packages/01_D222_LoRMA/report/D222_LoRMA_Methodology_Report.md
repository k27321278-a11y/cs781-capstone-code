# D222 + LoRMA — Phase 4 CJPE methodology

## Problem statement and data

Phase 4 Court Judgment Prediction predicts binary outcome 0/1 from legal case text. The preserved official inputs are Train/train.json, Train/dev.json and Test/test.json. Test labels are private. Original run records report 994 official dev labels and 1,517 test IDs; macro-F1 is the central metric.

## Architecture and mechanics

The 24-layer D222 causal decoder supplies contextual token states. Multiplicative low-rank updates adapt selected weights, while the evidence head learns token salience and ordered case-level context. Data flow: Legal case -> D222 own tokenizer/backbone -> Rank-8 multiplicative low-rank adaptation -> Token evidence pooling -> Ordered chunk BiGRU -> 2-class logits -> prediction.json.

## Training design and checkpoint selection

Rank-8 multiplicative low-rank adaptation. Executed v3 runner: `code/v3_t4x2.ipynb`, `code/runner/phase4_cjpe/kaggle_runner.py` with `published_v3_schedule.json`; the six jobs share `EvidenceCaseModel` in `code/runner/phase4_cjpe/models.py`.

## Exact run configuration and resource tradeoff

Experiment family d222; adaptation method lorma; original parent checkpoint SHA-256 08a737d002a71a90ad21f350e25a76f1e70eb172badf0671ff8818d96ea5c46f. Configuration: 256 tokens/chunk, 8 chunks/case, 8192 source characters considered, maximum 12000 training cases, 256 holdout cases per class. Optimizer configuration: backbone learning rate 0.0002, head learning rate 0.0003, gradient accumulation 8, max epochs 4, planned training hours 2.5. Recorded 213 optimizer updates and 1704 seen training examples; selected step 213. Selected checkpoint SHA-256 002e0dedacc5e8283ad8519c2ee6e6c9e3af15c83e3bf4cc2ee61178499b151a; parameters 226171274, trainable 4091402. Original run completed after nominal deadline: True.

## Measured results

Original Kaggle run: complete; 213 optimizer updates; selected step 213. Official 994-case dev macro-F1 **0.449472** at recorded threshold 0.284564; default 0.5 dev macro-F1 0.333333. Verified prediction ZIP SHA-256 `3be58f83f018fb99b1e491e0b69ff4507581b005de2f19d5fc52b8d01aa5af28`, exactly 1,517 IDs. User-provided Codabench screenshot: **0.4650** (finished submission).

## Evaluation mechanics

Official labeled dev cases 994; class supports [497, 497]; confusion matrix rows true 0/1, columns predicted 0/1: [[95, 402], [92, 405]]. At selected threshold 0.284564, macro-F1 0.449472; at threshold 0.5, macro-F1 0.333333. The report keeps these thresholds separate because threshold choice changes the binary predictions. The submitted ZIP has 1,517 unique test IDs and binary labels; no test labels were available to this run.

## Prediction contract and verification

The original prediction ZIP contains one root-level prediction.json with 1,517 unique official test IDs and integer labels restricted to 0 and 1. This package checks the ZIP CRC and SHA-256 against the archived run/verification record. Full model checkpoint tensors remain outside GitHub; their downloaded backup receipts and hashes live in the local Phase 4 study archives.

## Reproducibility and limits

The local dev score uses a threshold recorded in the run; inspect its calibration/selection record before treating it as independent model-selection evidence. A prepared model notebook is source scaffolding, while `v3_t4x2.ipynb` and the Kaggle run evidence identify the executed v3 lineage. Reports and companion notebook were created after training; original scores come from copied artifacts. Model weights and course data are preserved locally, not uploaded to GitHub.
