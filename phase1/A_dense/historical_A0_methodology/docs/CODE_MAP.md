# A0 Code-to-Methods Map — exact source spans

Source: `code/architecture_a_legalgpt.py`. All line numbers correspond to the **unaltered file** included in this package. Each row names the concrete API/function that implements a reproducible behavior. Function prose is an implementation summary; the Python code is the authoritative definition.

| Function or class | Source lines | Role and contract |
|---|---:|---|
| `ModelConfig` | 74–88 | Nominal architecture dimensions, RoPE base, dropout, activation checkpointing. validate enforces n_heads*d_head == d_model and uint16 vocab bound. |
| `DataConfig` | 92–103 | Official dataset sources, tokenizer/corpus caps, val split, source sampling and RNG defaults. |
| `TrainConfig` | 107–123 | Optimizer, training batch sizes, scheduler, clipping, evaluation/checkpoint intervals. |
| `GenerationConfig` | 127–135 | Greedy/default or optional sampling, temperature, top-k/top-p, repetition suppression, target words and limits. |
| `seed_everything` | 140–145 | Seed Python random, NumPy and torch CPU/CUDA RNGs; no global deterministic-algorithm mode. |
| `sha01` | 148–150 | Map first 8 bytes of SHA1 of UTF-8 input string to a float in [0,1). |
| `is_validation_doc` | 153–154 | Test hash of VAL::key against validation fraction. |
| `distributed_info` | 157–161 | Get WORLD_SIZE/RANK/LOCAL_RANK from environment. |
| `setup_distributed` | 164–172 | NCCL DDP initialization on CUDA if multiple ranks; otherwise CUDA/CPU device discovery. |
| `cleanup_distributed` | 175–178 | Distributed barrier and process group shutdown. |
| `rank0` | 181–182 | Check whether RANK is zero; used for logs/checkpoint I/O. |
| `extract_text` | 187–216 | Extract long legal-text field from supported JSON structures. |
| `extract_id` | 219–225 | Obtain ID field from one of six aliases, else stable fallback. |
| `legal_source_url` | 228–230 | URL of dataset file under Hugging Face main, with encoded path. |
| `stream_one_source` | 233–244 | Read official source JSON in streaming mode, shuffle buffer, filter <64-character documents, yield source/id/text. |
| `stream_natural` | 247–257 | Read combined phase_1 stream; shuffle/filter; yield source/id/text. |
| `stream_equal_mix` | 260–274 | Online equal-probability draw of one of four still-active source iterators; removes exhausted sources. |
| `iter_documents` | 277–289 | Dispatch to natural/equal source mode; on exception in equal mode warn/fallback to natural. |
| `tokenizer_paths` | 292–293 | Return saved tokenizer JSON and metadata JSON workdir paths. |
| `train_tokenizer` | 296–354 | Fit Hugging Face Byte-Level BPE on ≤100M non-val characters; min pair frequency 2; V target 32768; write tokenizer and metadata. |
| `load_tokenizer` | 357–363 | Load previously fitted tokenizer.json; error if absent. |
| `special_id` | 366–370 | Resolve PAD/BOS/EOS/UNK ID from saved tokenizer, error if missing. |
| `deterministic_window` | 373–382 | Choose one deterministic 640-word slice from a text containing ≥640 whitespace words. |
| `prepare_token_cache` | 385–483 | Stream documents, apply hashed split, encode manual BOS/EOS, write uint16 train/val binary caches and held-out generation examples; cache reuse if all files exist. |
| `resolve_hf_revision` | 486–491 | Best-effort retrieve dataset commit SHA via HfApi; None on error. |
| `RMSNorm` | 496–506 | Trainable scale, epsilon 1e-6; float32 RMS over last d features; no bias, no centering. |
| `RotaryEmbedding` | 509–530 | Precompute cos/sin position-frequency buffers for 32 adjacent Q/K feature pairs; offset-aware rotation, error beyond context. |
| `CausalMHA` | 533–578 | Fused bias-free QKV; split into 12 heads; RoPE Q/K; causal SDPA; cached single-token handling; bias-free output projection. |
| `SwiGLU` | 581–589 | Bias-free branches 768→2048; SiLU(W1x) elementwise multiplied by W3x; W2 projects 2048→768. |
| `TransformerBlock` | 592–609 | Apply RMSNorm-before-attention + residual; RMSNorm-before-SwiGLU + residual; return updated hidden and optional KV. |
| `LegalGPT` | 612–675 | Embedding, shared RoPE, 12 blocks, final RMSNorm, tied LM output, optional CE and caching; exact 110119680 nominal parameters. |
| `RandomMemmapBatcher` | 680–695 | Sample random windows of block_size+1 IDs; x=window[:-1], y=window[1:]; torch.long on requested device. |
| `build_optimizer` | 698–714 | Separate decay/no-decay by ndim; AdamW betas 0.9/0.95, eps1e-8; CUDA fused attempt/fallback. |
| `lr_for_step` | 717–724 | 2% step warmup then cosine to 10% of peak LR. |
| `set_lr` | 727–729 | Assign learning rate to both AdamW parameter groups. |
| `amp_dtype` | 732–737 | Select float32 CPU, bf16 CUDA when supported, else fp16 CUDA. |
| `autocast_context` | 740–743 | Autocast on CUDA; nullcontext CPU. |
| `evaluate_loss` | 746–756 | No-grad 32-batch validation CE loop; restores train mode; returns averaged loss. |
| `model_config_path` | 759–760 | Path to checkpoints/model_config.json. |
| `save_checkpoint` | 763–793 | Write model/optimizer/scaler/training metadata to latest.tmp.pt; atomic rename latest.pt; copy to best.pt if improved. |
| `load_training_checkpoint` | 796–806 | Load latest.pt model, optimizer, scaler and step/best_loss/tokens; no RNG state restoration. |
| `train` | 809–934 | Initialize model, optimizer, DDP/AMP, batchers; gradient accumulation, clip, optimizer updates, NLL evaluation, logs and saves. |
| `load_model_for_inference` | 939–953 | Load best.pt else latest.pt; rebuild architecture, tokenizer, and eval mode; disable checkpointing. |
| `apply_repetition_penalty` | 956–964 | On previously generated unique token IDs modify negative logits ×penalty, nonnegative ÷penalty. |
| `sample_token` | 967–994 | After repetition penalty, greedy argmax OR temperature/top-k/nucleus sampling (no beam). |
| `trim_to_words` | 997–1002 | Trim decoded text after the Nth regex whitespace word. |
| `word_count` | 1005–1006 | Count regex matches of non-whitespace spans. |
| `generate_continuation` | 1010–1060 | BOS+tokenized prompt, prefill KV, forbid special IDs, loop sampled token with cache until exactly 128 words or raise at hard token/context limit. |
| `load_val_examples` | 1065–1069 | Read data/val_continuations.json. |
| `compute_generation_metrics` | 1072–1087 | Compute stemmed ROUGE-L F1 and normalized SacreBLEU-4, optionally BERTScore with external scorer. |
| `evaluate_generation` | 1090–1115 | Generate held-out continuations for default 32 examples; write per-example and aggregate metrics JSON. |
| `download_test_json` | 1120–1127 | Fetch official test JSON into workdir/data. |
| `normalize_test_examples` | 1130–1155 | Normalize possible dict/list test schemas to (id,text) pairs; error if unsupported. |
| `make_submission` | 1158–1212 | Generate each official continuation, assert 128 words and complete IDs, produce ZIP with prediction.json as only root file. |
| `self_test` | 1217–1234 | Small 2-block forward CE and cached incremental decoding smoke test; does not train nominal A0. |
| `build_parser` | 1239–1287 | Expose modes and CLI options/defaults; several dataclass internals not configurable by CLI. |
| `configs_from_args` | 1290–1334 | Translate CLI arguments into four config dataclasses. |
| `main` | 1337–1373 | Create workdir, seed, persist run_config, dispatch modes; overwrites run_config on each invocation. |

## Reimplementation-critical source relationships

- `LegalGPT.__init__` instantiates a *single* `RotaryEmbedding` that all 12 `TransformerBlock` objects share (nontrainable buffers); it separately instantiates all learnable weights for each block.
- `LegalGPT.forward` returns `(logits, optional_loss, optional_layerwise_cache)` and does not apply final softmax. `torch.nn.functional.cross_entropy` takes its raw logits.
- `CausalMHA.forward` stores K and V **after RoPE for K**, not unrotated K. It concatenates past K/V before SDPA and permits single-token cached query only.
- `TransformerBlock.forward` is Pre-RMSNorm in both sublayers and uses two residual additions.
- `RandomMemmapBatcher` samples token-ID windows from one packed binary stream, not document-aligned 512/128 text pairs.
- `evaluate_loss` is the source for checkpoint selection; `evaluate_generation` computes separate held-out continuation metrics and never controls saving `best.pt`.
- `generate_continuation` is responsible for fixed **128-word** output; `sample_token` is responsible for selecting one **BPE token ID** at a time.
- `train_tokenizer` trains a BPE vocabulary; `sample_token` is unrelated Transformer-generation decoding.
- `make_submission` constructs the separate **Codabench prediction ZIP**. The final Hello IITK code/report ZIP requires a different directory layout.

## Audit note

A source-level claim is reproducible only when its code path and initial configuration are both known; optional paths and default paths must be described separately. No training output files or checkpoint from a completed full-size run were part of the supplied original source packages.
