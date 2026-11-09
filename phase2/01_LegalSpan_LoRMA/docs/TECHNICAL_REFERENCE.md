# LegalSpan with LoRMA+ — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

The 12-layer causal LegalSpan backbone remains frozen while rank-8 multiplicative adapters modify selected projections. A character-level bidirectional GRU and 25-way BIO classifier align token features with exact character offsets.

`Legal text → Own byte BPE → 12-layer LegalSpan → LoRMA+ projections → Character BiGRU → 25 BIO logits → Span decoder`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| `vocab_size` | `32768` |
| `max_seq_len` | `8192` |
| `n_layers` | `12` |
| `d_model` | `768` |
| `n_heads` | `12` |
| `d_head` | `64` |
| `d_ff` | `2048` |
| `rope_theta` | `10000.0` |
| `dropout` | `0.0` |
| `gradient_checkpointing` | `True` |
| `route_layers` | `4` |
| `route_dim` | `128` |
| `route_mode` | `full` |
| `use_roles` | `True` |
| `use_continuity` | `True` |
| `use_fusion` | `True` |
| `use_copy` | `True` |

## Phase 2 optimization [RUN/CODE]

The recorded method is **LoRMA+, rank 8**. Run-specific settings: `{"name": "01_legalspan_lorma", "family": "legalspan", "method": "lorma", "encoding": "own", "rank": 8, "alpha": 32, "base_lr": 0.0002, "head_lr": 0.0005, "checkpoint": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_legalspan_v1/experiment/full/checkpoints/latest.pt", "tokenizer": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_legalspan_v1/prepared_v2/tokenizer.json", "submission_name": "UPLOAD_Phase2_01_LegalSpan_LORMA.zip"}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.411641173**. The archived prediction ZIP has 35 official IDs and SHA-256 `b26d7dc4b8ac8281ef8a3a9f3d570e32668a2fc6d9088434bd15a101912d0cd6`. The selected checkpoint SHA-256 is `609c3b97603769ed8035900cad20121fd3160a5e8cd36ae1d2ba32a7a9ef9819`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
