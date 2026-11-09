# Exact-span two-of-four consensus

## Abstract
This is a prediction-level ensemble, not a newly trained neural architecture. A span survives if at least two of the first four models output the same start, end, and label; conflicting overlaps are resolved by vote count then span length.

## Method


This is a prediction-level ensemble, not a newly trained neural architecture. A span survives if at least two of the first four models output the same start, end, and label; conflicting overlaps are resolved by vote count then span length.

`LegalSpan → B1 → C1 → A0 → Exact span votes → Overlap resolution → Prediction JSON`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| Config | See the four source models |

## Phase 2 optimization [RUN/CODE]

The recorded method is **Two-of-four exact span vote**. Run-specific settings: `{}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.463066765; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
