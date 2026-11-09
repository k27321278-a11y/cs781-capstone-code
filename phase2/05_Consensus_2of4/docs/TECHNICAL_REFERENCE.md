# Exact-span two-of-four consensus — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

This is a prediction-level ensemble, not a newly trained neural architecture. A span survives if at least two of the first four models output the same start, end, and label; conflicting overlaps are resolved by vote count then span length.

`LegalSpan → B1 → C1 → A0 → Exact span votes → Overlap resolution → Prediction JSON`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| Config | See the four source models |

## Phase 2 optimization [RUN/CODE]

The recorded method is **Two-of-four exact span vote**. Run-specific settings: `{}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.463066765**. The archived prediction ZIP has 35 official IDs and SHA-256 `c7ebcdda565008949c9df37578610277161520173db5ef90a4b34e533c0d36f0`. The selected checkpoint SHA-256 is `None`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export; rule selected on validation.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
