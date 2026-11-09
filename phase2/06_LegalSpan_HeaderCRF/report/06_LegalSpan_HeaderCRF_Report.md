# LegalSpan header attention and CRF

## Abstract
The selected LegalSpan LoRMA NER checkpoint is merged into the backbone, then fully tuned. A header-character BiGRU and four-head attention inform each character; a 25-tag CRF and boundary loss encourage coherent spans.

## Method


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



## Results and limits
Local strict held-out macro-F1 0.411516001; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
