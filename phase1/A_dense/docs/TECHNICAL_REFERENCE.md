# A0 dense causal Transformer — technical reference

## Evidence classes

`[CODE]` means inspected recovered implementation. `[ARCHIVED]` means a copied October 6 run artifact. `[SCREENSHOT]` means historically reported Codabench score, not fresh leaderboard retrieval. `[INFERENCE]` is scientific interpretation.

## Problem [CODE]

Train a legal language model from four official text sources (Acts, High Court, Supreme Court, Tribunal) and generate exactly 128 words conditioned on each 512-word prompt. The major measured generation metric is BERTScore F1, alongside ROUGE-L and BLEU-4 in local experiments. Different policies and checkpoints must be compared on the same split.

## Computation graph [CODE]

A 12-block decoder-only Transformer predicts the next subword token from a causal context. Width 768, 12 attention heads of 64, 2048-wide SwiGLU feed-forward layers, RMSNorm, rotary positions, and tied input/output embeddings make the 110.1M-parameter graph.

`512-word prompt → Library-trained BPE → 12 causal blocks → RMSNorm / RoPE / SwiGLU → Vocabulary logits → 128-word continuation`

## Exact model configuration [ARCHIVED]

```json
{
  "vocab_size": 32768,
  "max_seq_len": 2048,
  "n_layers": 12,
  "d_model": 768,
  "n_heads": 12,
  "d_head": 64,
  "d_ff": 2048,
  "rope_theta": 10000.0,
  "dropout": 0.0,
  "gradient_checkpointing": true
}
```

## Mechanics [CODE]

The tokenizer maps bytes or strings to one of 32,768 learned IDs. An embedding produces width-dimensional vectors. Attention forms Q/K/V projections, masks unavailable future positions where causal, and mixes values by normalized compatibility. Rotary position phases rotate Q and K, RMSNorm controls residual scale, and SwiGLU gates the feed-forward branch. Autoregressive decoding samples or selects one token at a time and checks word count. LegalSpan adds a prompt route and copy mixture, B adds source encoder memory plus cross-attention, C adds an FP32 GRU, D222 groups Q heads over fewer K/V heads and alternates local/global context.

## Data, training and validation [ARCHIVED/CODE]

Inspect `code/configs/a.json` for the exact optimizer, warmup, batch, context and decoding settings. The original complete archive contains 6 copied small evidence files here and large checkpoint/tokenizer/data bytes separately. Train-only fitting, held-out local evaluation, checkpoint choice, and final export are distinct stages. A model with lower token loss may still produce worse continuations under a chosen decoding policy.

## Result [SCREENSHOT or ARCHIVED]

0.8304 (historical Codabench screenshot). `evidence/archived_metric_index.json` lists 0 local metric files extracted from the original archive. The archive itself contains no original executed notebook. The illustrated and code-review notebooks here were assembled later from original source; saved outputs are not forged.

## Reproduction boundary

The complete original archive is `C:\Users\Iwill\Documents\Codex\2026-10-06\referenced-chatgpt-conversation-this-is-an\outputs\CS781_A0_Complete_20261006.zip`. It contains large selected weights and additional run artifacts. This compact package avoids copying official source data or multi-gigabyte checkpoints. Update historical GPU-machine paths before running code on another computer.
