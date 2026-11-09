# A0 distilled from B1 NER — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

A 12-layer causal A0 student is fully tuned. The completed B1 LoRA NER model supplies temperature-2 teacher probabilities on training windows only; the loss mixes hard-label cross-entropy and teacher KL equally.

`Legal text → A0 tokenizer → 12-layer A0 student → B1 teacher on train → CE + temperature-2 KL → Character BiGRU → Span decoder`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| `vocab_size` | `32768` |
| `max_seq_len` | `2048` |
| `n_layers` | `12` |
| `d_model` | `768` |
| `n_heads` | `12` |
| `d_head` | `64` |
| `d_ff` | `2048` |
| `rope_theta` | `10000.0` |
| `dropout` | `0.0` |
| `gradient_checkpointing` | `True` |

## Phase 2 optimization [RUN/CODE]

The recorded method is **Supervised + teacher KL**. Run-specific settings: `{"name": "04_a0_distilled", "family": "a0", "method": "kd", "encoding": "a0", "base_lr": 2e-05, "head_lr": 0.0003, "teacher": "02_b1_lora", "checkpoint": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_phase1_5pass_20261003/checkpoints/latest.pt", "tokenizer": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source/prepared/tokenizer.json", "submission_name": "UPLOAD_Phase2_04_A0_Distilled.zip"}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.409578526**. The archived prediction ZIP has 35 official IDs and SHA-256 `6b2ddb633ea55ad837549ffe06c3a26239204c92cfd133986f6aa9856daf144e`. The selected checkpoint SHA-256 is `2e605bafac82bfab0bc6d584732c2923e079f1038a3951d4f536d299740fc0d3`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
