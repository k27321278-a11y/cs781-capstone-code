# A0 distilled from B1 NER

## Abstract
A 12-layer causal A0 student is fully tuned. The completed B1 LoRA NER model supplies temperature-2 teacher probabilities on training windows only; the loss mixes hard-label cross-entropy and teacher KL equally.

## Method


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



## Results and limits
Local strict held-out macro-F1 0.409578526; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
