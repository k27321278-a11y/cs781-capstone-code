#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
[[ -f .env_runtime ]] && source .env_runtime

# GPU may be supplied: GPU=1 ./train_A0.sh
source ./select_gpu.sh

WORKDIR="${WORKDIR:-$ROOT/runs/A0_dense_flash}"
mkdir -p "$WORKDIR/logs"

echo "Training workdir: $WORKDIR"
echo "Physical GPU requested/selected: ${GPU:-auto}; visible device inside PyTorch will be cuda:0"

python -u architecture_a_legalgpt.py --mode train \
  --workdir "$WORKDIR" \
  --max_seq_len 2048 \
  --micro_batch_size 1 \
  --grad_accum_steps 32 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --target_train_tokens 200000000 \
  --eval_every 100 \
  --eval_batches 32 \
  --save_every 100 \
  --log_every 10 2>&1 | tee -a "$WORKDIR/logs/train.log"
