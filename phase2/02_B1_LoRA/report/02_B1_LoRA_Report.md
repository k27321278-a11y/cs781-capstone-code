# B1 encoder-decoder with LoRA

## Abstract
The 6+6-layer B1 encoder-decoder supplies contextual decoder states. Additive low-rank updates train selected linear projections while the pretrained backbone weights remain frozen; a character BiGRU produces BIO tags.

## Method


The 6+6-layer B1 encoder-decoder supplies contextual decoder states. Additive low-rank updates train selected linear projections while the pretrained backbone weights remain frozen; a character BiGRU produces BIO tags.

`Legal text → Own byte BPE → 6-layer encoder → 6-layer decoder + LoRA → Character BiGRU → 25 BIO logits → Span decoder`

## Backbone configuration [RUN]

| Field | Value |
|---|---|
| `vocab_size` | `32768` |
| `max_seq_len` | `8192` |
| `n_encoder_layers` | `6` |
| `n_decoder_layers` | `6` |
| `d_model` | `768` |
| `n_heads` | `12` |
| `d_head` | `64` |
| `d_ff` | `2048` |
| `rope_theta` | `10000.0` |
| `dropout` | `0.0` |
| `gradient_checkpointing` | `True` |
| `pad_token_id` | `0` |

## Phase 2 optimization [RUN/CODE]

The recorded method is **LoRA, rank 8**. Run-specific settings: `{"name": "02_b1_lora", "family": "b1", "method": "lora", "encoding": "own", "rank": 8, "alpha": 32, "base_lr": 0.0002, "head_lr": 0.0005, "checkpoint": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006/b1_improved/checkpoints/latest.pt", "tokenizer": "/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006/tokenizer.json", "submission_name": "UPLOAD_Phase2_02_B1_LoRA.zip"}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.386274524; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
