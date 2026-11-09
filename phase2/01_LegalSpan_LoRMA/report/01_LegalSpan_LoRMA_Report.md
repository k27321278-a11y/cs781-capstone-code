# LegalSpan with LoRMA+

## Abstract
The 12-layer causal LegalSpan backbone remains frozen while rank-8 multiplicative adapters modify selected projections. A character-level bidirectional GRU and 25-way BIO classifier align token features with exact character offsets.

## Method


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



## Results and limits
Local strict held-out macro-F1 0.411641173; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
