# CS781 Architecture B: GPU1 handoff

## What is in the bundle

The ZIP adds Architecture B's bidirectional prompt encoder, causal Transformer decoder with prompt cross-attention, shared BPE embedding/output table, B-specific batching/training/generation/CLI code, configuration, focused tests, and a guarded server runner. The default graph is expected to contain **124,280,832 unique parameters**; a meta-device test checks this without allocating the model's full CPU weights. It does not overwrite A's trainer, model, CLI, config, checkpoints, or run directory.

The substantive target budget is **209,914,685 supervised target tokens**, matching the target reported for A. B initializes its own weights from scratch; it reuses only the official prepared data, split, tokenizer IDs, and shared pipeline interfaces.

## Verification state

Python syntax and the B JSON config parse locally. I could not execute the PyTorch tests or benchmark here because this local Python runtime has no `torch` or `pytest`. I also could not access the connected server from this task: `ssh stud1` failed because this session cannot resolve that host, and no VS Code terminal is attached to the task. Therefore B has **not** been installed on the server or launched by me. A's running process and the paused jobs were not touched.

The included server runner executes the B-focused tests, checks that `/media/hdd` is mounted, checks for another B process, requires at least 8 GiB free on GPU1, runs a 125-step CUDA sizing benchmark, then starts the full B training run only if the benchmark succeeds. It does not stop or signal other processes.

## Transfer and launch

Save `CS781_B_GPU1_HANDOFF.zip` on the server or copy it into the VS Code SSH workspace. In the server terminal:

```bash
cd /home/stud1/cs781_phase1_v2
unzip -o /path/to/CS781_B_GPU1_HANDOFF.zip -d .
chmod +x scripts/run_b_gpu1.sh
LOG=/home/stud1/cs781_phase1_v2/b_gpu1_supervisor_20261003.log
nohup bash scripts/run_b_gpu1.sh >"$LOG" 2>&1 </dev/null &
echo $! >/home/stud1/cs781_phase1_v2/b_gpu1_supervisor_20261003.pid
```

This uses the existing Python environment at `/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python` and the prepared data under `official_100m_chars_per_source/prepared`. `CUDA_VISIBLE_DEVICES=1` binds B to physical GPU1. The run uses a distinct HDD directory and does not modify A's run.

If the supervisor stops before training, inspect its log and the benchmark log; do not delete its run directory or start a second copy. It refuses a duplicate B process and refuses to overwrite existing benchmark/checkpoint artifacts.

## Live status

```bash
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2
RUN="$ART/b_phase1_5pass_20261003"
date -u
pgrep -u "$USER" -af 'cs781.cli_b' || true
tail -n 2 "$RUN/metrics.jsonl" 2>/dev/null
cat "$RUN/training_status.json" 2>/dev/null
ls -lh "$RUN/checkpoints/latest.pt" 2>/dev/null
nvidia-smi -i 1 --query-gpu=memory.used,memory.free,utilization.gpu --format=csv
```

`training_status.json` must say `complete`, and its checkpoint hash/state must match `checkpoints/latest.pt`, before the official prediction command accepts that checkpoint.

## Prediction ZIP after training completes

The runner intentionally stops after training. Once the existing official-data network route is confirmed reachable, generate and validate the official 1,000-item prediction ZIP:

```bash
cd /home/stud1/cs781_phase1_v2
export PYTHONPATH=.runtime:.test-deps:.
export CUDA_VISIBLE_DEVICES=1
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2
DATA="$ART/official_100m_chars_per_source/prepared"
RUN="$ART/b_phase1_5pass_20261003"
"$PY" -u -m cs781.cli_b predict-official-test \
  --data "$DATA" \
  --checkpoint "$RUN/checkpoints/latest.pt" \
  --output "$ART/b_phase1_5pass_20261003_submission.zip" \
  --config configs/b.json --device cuda
```

The official command requires a completed non-synthetic run, matching code/data/tokenizer provenance, all 1,000 official IDs, exactly 512 prompt words and 128 continuation words, and writes `prediction.json` at the ZIP root. It fails rather than fabricate or omit outputs. Real dev scoring still requires the separately pinned BERTScore cache described in the main handoff.

## Design references and disclosure

The B computation graph follows the approved project specification. Background references for the borrowed components are: [Transformer](https://arxiv.org/abs/1706.03762), [RoPE](https://arxiv.org/abs/2104.09864), [SwiGLU](https://arxiv.org/abs/2002.05202), and [RMSNorm](https://arxiv.org/abs/1910.07467). The student should retain these citations in the capstone report and disclose AI coding assistance according to the course rules.
