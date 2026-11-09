#!/usr/bin/env bash
set -euo pipefail
OUT=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase2_20261011/runs
date -Is
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
if [[ -f "$OUT/queue.pid" ]]; then ps -p "$(cat "$OUT/queue.pid")" -o pid,ppid,stat,etime,args; fi
cat "$OUT/queue_status.json" 2>/dev/null || true
for f in "$OUT"/*/status.json; do [[ -f "$f" ]] && cat "$f" && printf '\n'; done
tail -5 "$OUT/queue.log"
