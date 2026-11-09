#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/stud1/cs781_phase1_cde_v3
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2
RUN="$ART/c_phase1_10pass_20261003"
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.runtime:.test-deps:.
mkdir -p "$ART"
exec 9>"$ART/c_gpu0_supervisor.lock"
flock -n 9 || { echo 'C supervisor is already active'; exit 3; }
if pgrep -af '[c]s781.cli_c train' >/dev/null; then
    echo 'An existing C training process is visible; refusing a duplicate'
    exit 3
fi
if [[ -f "$RUN/checkpoints/latest.pt" || -e "$RUN/metrics.jsonl" ]]; then
    echo 'C run already exists; inspect state and explicitly resume rather than overwrite'
    exit 3
fi
FREE=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( FREE < 6000 )); then
    echo "GPU0 has only $FREE MiB free; leave existing jobs untouched"
    exit 4
fi
ps -ww -p 284634,469837,469841,497086,524833,526131,526372 -o pid,stat,args || true
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo 'Starting C on physical GPU0, fresh weights, 419829370 supervised target tokens'
exec "$PY" -u -m cs781.cli_c train --data "$ART/official_100m_chars_per_source/prepared" \
    --run-dir "$RUN" --config configs/c.json --device cuda
