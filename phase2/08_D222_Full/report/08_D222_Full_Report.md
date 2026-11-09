# D222 with full NER tuning

## Abstract
D222 is a 24-layer, width-896 causal decoder with grouped-query attention: 14 query heads share two key/value heads. Every fourth layer is global; others use a 512-token local window. The NER head adds a character BiGRU, boundary logits, and CRF.

## Method


D222 is a 24-layer, width-896 causal decoder with grouped-query attention: 14 query heads share two key/value heads. Every fourth layer is global; others use a 512-token local window. The NER head adds a character BiGRU, boundary logits, and CRF.

`Legal text → Own byte BPE → 24-layer D222 GQA → Local/global attention → Character BiGRU → CRF + boundaries → Span export`

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

The recorded method is **Full fine-tuning**. Run-specific settings: `{"name": "full_lr2e5", "method": "full", "backbone_lr": 2e-05, "head_lr": 0.0005, "dropout": 0.1}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.400950487; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
