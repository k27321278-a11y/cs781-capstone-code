# LegalSpan header attention and CRF — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

The selected LegalSpan LoRMA NER checkpoint is merged into the backbone, then fully tuned. A header-character BiGRU and four-head attention inform each character; a 25-tag CRF and boundary loss encourage coherent spans.

`Legal text + header → Merged LegalSpan → Header BiGRU → 4-head cross-attention → BIO emissions + CRF → Boundary loss → Span decoder`

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

The recorded method is **Full tuning + header attention + CRF**. Run-specific settings: `{"name": "full_header_crf", "method": "full", "backbone_lr": 1e-05, "head_lr": 0.0002, "dropout": 0.1}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.411516001**. The archived prediction ZIP has 35 official IDs and SHA-256 `54b7c252578672a5373af5a98d24fd749f007ea5515ad8967c9c08e7c0ffd1f6`. The selected checkpoint SHA-256 is `2cb1c2dbb702b9ef85b9bfab1d160db2fcbb390542da1baf9e043c6642298f80`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
