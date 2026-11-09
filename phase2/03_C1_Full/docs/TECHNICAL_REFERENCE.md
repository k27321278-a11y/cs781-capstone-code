# C1 recurrent hybrid, full tuning — technical reference

## Provenance labels

- **[CODE]** describes an inspected source implementation.
- **[RUN]** is copied from the completed run record.
- **[LOCAL]** is the 12-document held-out result, not a Codabench score.
- **[INTERPRETATION]** explains a plausible mechanism without claiming measured causality.

## Task and representation [CODE]

The task is 12-class legal named entity recognition with exact `[start,end,label]` character spans. The first four models and D222 variants use two labeled training folds (70 documents) with a deterministic 58/12 train/validation split; official test has 35 IDs. The basic tagging space has `1 + 2*12 = 25` labels, outside and BIO pairs. Original character offsets are preserved through tokenization and window stitching. Empty test documents receive an empty list.

## Architecture [CODE]

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

## Output mapping [CODE]

The token backbone produces contextual vectors. A character-to-token map gathers one vector per original source character; a learned 32-dimensional character embedding provides spelling cues. A bidirectional 128-unit GRU emits 256-dimensional contextual character features. BIO logits and constrained decoding recover exact source offsets. D222 adds a learned CRF and boundary objective; HeaderCRF adds header attention; RoleSpan first proposes spans then predicts legal roles. The consensus operates only on exact span triples.

## Observed result [LOCAL]

Strict held-out macro-F1: **0.357207372**. The archived prediction ZIP has 35 official IDs and SHA-256 `090088e3e034fbcfed3267d3bf46ef9c6a603110407839546b23b943d9fffe54`. The selected checkpoint SHA-256 is `1e5df7c9cca24f1d4113618e0d9e1f2cdee85db02b931fea7a4d1c9105f012a7`. The existence of a valid export does not prove a Codabench upload or hidden score. Status: Pre-deadline export.

## Limitations [INTERPRETATION]

This is a small held-out set. The consensus voting rule was selected using that validation set, so its 0.4631 local score is exploratory. D222 100M and 200M exports completed after the original deadline. Large weights, tokenizers, and official data are retained in the named original archive and intentionally omitted from this compact explanatory package.
