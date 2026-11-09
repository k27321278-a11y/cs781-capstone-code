#!/usr/bin/env bash
set -euo pipefail
CODE=/home/stud1/cs781_phase2_20261011
OUT=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase2_20261011/runs
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
mkdir -p "$OUT"
if [[ -f "$OUT/queue.pid" ]]; then
  pid=$(cat "$OUT/queue.pid")
  if kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'lner.train'; then
    echo "ALREADY_RUNNING PID=$pid"; exit 0
  fi
fi
uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -Fq "$uuid"; then
  echo 'GPU1 has an existing compute process. Launch deferred.' >&2; exit 2
fi
free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free < 11000 )); then echo 'GPU1 has insufficient free memory.' >&2; exit 2; fi
cd "$CODE"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$CODE/backbones:$CODE"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
nohup "$PY" -u -m lner.train --config configs/queue.json > "$OUT/queue.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$OUT/queue.pid"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then tail -30 "$OUT/queue.log"; exit 1; fi
echo "STARTED PID=$pid PHYSICAL_GPU=1 UUID=$uuid"
