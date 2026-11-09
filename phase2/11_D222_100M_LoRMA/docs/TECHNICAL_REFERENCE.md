# D222 100M checkpoint + LoRMA — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

An intermediate D222 pretraining snapshot at about 100M supervised tokens initializes the same rank-8 LoRMA NER model. This is a checkpoint-depth study, not a different D222 block architecture.

`D222 100M LM → Own byte BPE → 24-layer GQA → Rank-8 LoRMA+ → Character BiGRU → CRF + boundaries → Span export`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| `vocab_size` | `32768` |
| `layers` | `24` |
| `width` | `896` |
| `query_heads` | `14` |
| `kv_heads` | `2` |
| `head_dim` | `64` |
| `ff_width` | `2304` |
| `context` | `2048` |
| `local_window` | `512` |
| `global_every` | `4` |
| `query_tile` | `128` |
| `dropout` | `0.0` |
| `rope_theta` | `10000.0` |
| `checkpointing` | `True` |

## Phase 2 optimization [RUN/CODE]

The recorded method is **100M-token LM checkpoint, LoRMA+**. Run-specific settings: `{"name": "lorma_100M", "method": "lorma", "backbone_lr": 0.0002, "head_lr": 0.0005, "dropout": 0.1}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.420073844**. The archived prediction ZIP has 35 official IDs and SHA-256 `f24b24f5d99485baeda6ea7b4b3d3cd91b679a4aab00992c46ae5348c5adefa7`. The selected checkpoint SHA-256 is `5121db8c40b9f27ffa02ace404bbcbb182f7859a74d69808c8bdaa3065624784`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Completed after original Phase 2 deadline.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
