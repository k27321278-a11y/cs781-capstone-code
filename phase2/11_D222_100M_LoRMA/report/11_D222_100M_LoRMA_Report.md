# D222 100M checkpoint + LoRMA

## Abstract
An intermediate D222 pretraining snapshot at about 100M supervised tokens initializes the same rank-8 LoRMA NER model. This is a checkpoint-depth study, not a different D222 block architecture.

## Method


An intermediate D222 pretraining snapshot at about 100M supervised tokens initializes the same rank-8 LoRMA NER model. This is a checkpoint-depth study, not a different D222 block architecture.

`D222 100M LM → Own byte BPE → 24-layer GQA → Rank-8 LoRMA+ → Character BiGRU → CRF + boundaries → Span export`

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

The recorded method is **100M-token LM checkpoint, LoRMA+**. Run-specific settings: `{"name": "lorma_100M", "method": "lorma", "backbone_lr": 0.0002, "head_lr": 0.0005, "dropout": 0.1}`. The common first-four runner windows at most 1600 characters, with 240-character overlap and at most 768 tokens per window; it accumulates 8 windows, uses AdamW and cosine decay with warmup, weighted character cross-entropy and checkpoint selection by held-out strict macro-F1. A0 also has an equal-weight teacher KL term with temperature 2 on train windows. D222 and extended LegalSpan models have their own NER objectives and schedules in the source files; do not substitute the common runner's hyperparameters for them.



## Results and limits
Local strict held-out macro-F1 0.420073844; no verified hidden score. The source snapshot, selected-checkpoint identity, and prediction hash are recorded in this package.
