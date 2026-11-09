# LegalSpan coarse spans and legal roles

## Abstract
This warm-started LegalSpan variant first proposes generic spans, then classifies each span into one of six legal roles from boundary, pooled, header, and length features. It also contains a repeated-name consistency mechanism.

## Method


This warm-started LegalSpan variant first proposes generic spans, then classifies each span into one of six legal roles from boundary, pooled, header, and length features. It also contains a repeated-name consistency mechanism.

`Legal text + header → Warm-start LegalSpan → Coarse BIO tags → CRF span proposal → Span feature pooling → Six legal roles → Consistency + export`

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

The recorded method is **Coarse span + six-role classifier**. Run-specific settings: `{"name": "role_span", "method": "full", "backbone_lr": 5e-06, "head_lr": 0.0003, "dropout": 0.1}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.408720599; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
