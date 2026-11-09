# C1 recurrent hybrid, full tuning

## Abstract
The C1 backbone combines a 10-layer encoder with a four-layer GRU decoder and cross-attention. Phase 2 updates all model parameters, then uses the shared character BiGRU and BIO classifier.

## Method


The C1 backbone combines a 10-layer encoder with a four-layer GRU decoder and cross-attention. Phase 2 updates all model parameters, then uses the shared character BiGRU and BIO classifier.

`Legal text → Own byte BPE → 10-layer encoder → 4-layer GRU + cross-attn → Character BiGRU → 25 BIO logits → Span decoder`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| `vocab_size` | `32768` |
| `max_seq_len` | `2048` |
| `max_prompt_len` | `8192` |
| `n_encoder_layers` | `10` |
| `n_gru_layers` | `4` |
| `d_model` | `768` |
| `n_heads` | `12` |
| `d_head` | `64` |
| `d_ff` | `2048` |
| `rope_theta` | `10000.0` |
| `dropout` | `0.0` |
| `gradient_checkpointing` | `True` |
| `pad_token_id` | `0` |

## Phase 2 optimization [RUN/CODE]

The recorded method is **Full fine-tuning**. Run-specific settings: `{"name": "03_c1_full", "family": "c1", "method": "full", "encoding": "own", "base_lr": 2e-05, "head_lr": 0.0003, "checkpoint": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006/c1_scratch/checkpoints/latest.pt", "tokenizer": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006/tokenizer.json", "submission_name": "UPLOAD_Phase2_03_C1_Full.zip"}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.357207372; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
