# D222 with full NER tuning — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

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

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.400950487**. The archived prediction ZIP has 35 official IDs and SHA-256 `6aaddc32ca90f506b258fba8d4e569b8aa9589252b14aa59cf9904dc9e9234ed`. The selected checkpoint SHA-256 is `773f9d8a7fb08f61e684956ed890d2285b163a6d7fb5d8a782aa50146613b1f1`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
