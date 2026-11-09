# CS781 Phase 1 — Architecture A — RTX server package

This is the server-ready version of Architecture A: a from-scratch ~110M parameter legal decoder LM.

## Model

- 12 decoder blocks
- d_model 768
- 12 full MHA heads x 64 dimensions
- SwiGLU d_ff 2048
- RoPE
- Pre-RMSNorm
- tied embedding/LM head
- PyTorch SDPA (Flash kernel is used when supported)
- exact nominal params at vocab 32768: 110,119,680

## Important while FeelWM is still running

Do **not** start this training on a GPU that is already substantially occupied by FeelWM. Run:

```bash
./rtx_preflight.sh
```

If another RTX is free, select it explicitly, e.g.:

```bash
GPU=1 ./train_A0.sh
```

If there is only one GPU and it is busy, run data preparation now (CPU/network/disk) and start training after the GPU is free.

## 1. Extract and enter folder

```bash
unzip CS781_Phase1_A_RTX_Server.zip
cd CS781_Phase1_A_RTX_Server
```

## 2. Preflight

```bash
./rtx_preflight.sh
```

## 3. Create an isolated Python environment

```bash
./setup_env.sh
source .venv/bin/activate
source .env_runtime
```

All Hugging Face, Torch and temporary caches are kept under this project folder.

## 4. Prepare data/tokenizer (can run while GPU is busy)

```bash
./prepare_A0.sh
```

This trains the 32K byte-level legal BPE and builds approximately 200M train + 5M validation token caches using only the official Phase-1 corpus.

## 5. Train on one free RTX

Auto-select the GPU with the most free VRAM:

```bash
./train_A0.sh
```

Or pin physical GPU 1:

```bash
GPU=1 ./train_A0.sh
```

The launcher refuses to start if selected free VRAM is below `MIN_FREE_MIB` (default 9000 MiB). Override only if you have measured that it is safe:

```bash
MIN_FREE_MIB=7500 GPU=1 ./train_A0.sh
```

Checkpoint/resume is automatic from:

```text
runs/A0_dense_flash/checkpoints/latest.pt
```

Rerunning the same command resumes training.

### Keep the run alive after SSH disconnects

Use tmux:

```bash
tmux new -s cs781_A0
GPU=1 ./train_A0.sh
```

Detach with `Ctrl-b d`. Reattach later:

```bash
tmux attach -t cs781_A0
```

## 6. Monitor

```bash
./watch_A0.sh
```

or:

```bash
tail -f runs/A0_dense_flash/logs/train.log
```

## 7. Local held-out generation evaluation

```bash
GPU=1 ./eval_A0.sh
```

This reports ROUGE-L and BLEU-4 on held-out 512-word -> 128-word examples. The authoritative BERTScore is obtained from Codabench.

## 8. Generate the Codabench submission

```bash
GPU=1 ./submit_A0.sh
```

Upload **only**:

```text
runs/A0_dense_flash/submission/greedy/submission.zip
```

Its structure is exactly:

```text
submission.zip
└── prediction.json
```

`prediction.json` contains every official test ID and exactly a 128-word continuation, without the original prompt.

## A0 research settings

- train tokens: 200M
- val tokens: 5M
- context: 2048 BPE tokens
- optimizer: AdamW
- peak LR: 3e-4
- betas: (0.9, 0.95)
- weight decay: 0.1
- gradient clipping: 1.0
- warmup: 2%
- cosine decay to 10% of peak LR
- microbatch: 1
- grad accumulation: 32
- gradient checkpointing: enabled

A0 is intentionally the controlled baseline. After its first Codabench score we optimize decoding/data/attention choices one variable at a time before moving to PrefixLM E.
