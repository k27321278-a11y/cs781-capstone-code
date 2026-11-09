#!/usr/bin/env bash
set -euo pipefail
CODE=/home/stud1/cs781_d222_20261011
OUT=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_d222_20261011
PREVIOUS=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase2_20261011/runs/04_a0_distilled/complete.json
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
mkdir -p "$OUT"
"$PY" -c 'import json,hashlib,sys; r=json.load(open(sys.argv[1])); h=hashlib.sha256(open(r["submission"],"rb").read()).hexdigest(); assert r["phase"]=="complete" and h==r["sha256"]' "$PREVIOUS"
if [[ -f "$OUT/queue.pid" ]]; then
 pid=$(cat "$OUT/queue.pid")
 if kill -0 "$pid" 2>/dev/null && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'legal_d.pipeline'; then
  echo "ALREADY_RUNNING PID=$pid"; exit 0
 fi
fi
uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -Fq "$uuid"; then
 echo 'GPU1 is occupied; no shared process will be stopped.' >&2; exit 2
fi
cd "$CODE"
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$CODE:$CODE/vendor:$CODE/vendor/backbones" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
if [[ ! -f "$OUT/preflight.json" ]]; then echo 'Run the documented preflight before launch.' >&2; exit 2; fi
"$PY" -c 'import json,torch; from pathlib import Path; from legal_d.common import source_hashes; from legal_d.data import verify_prepared; c=json.load(open("configs/d222.json")); r=json.load(open(Path(c["output_root"])/"preflight.json")); assert r["fingerprint"]=={"config":c,"data":verify_prepared(c),"source_hashes":source_hashes()}; assert r["causality"]=="PASS" and r["finite_real_gradients"] and r["ner_crf_finite_gradient"] and r["empty_document"]=="PASS" and r["gpu"]==torch.cuda.get_device_name(0) and r["cuda_visible_devices"]=="1"'
log="$OUT/queue.$(date -u +%Y%m%dT%H%M%S).log"
nohup "$PY" -u -m legal_d.pipeline --config configs/d222.json > "$log" 2>&1 < /dev/null &
pid=$!; printf '%s\n' "$pid" > "$OUT/queue.pid"; printf '%s\n' "$log" > "$OUT/queue_log_path.txt"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then tail -20 "$log"; exit 1; fi
echo "STARTED PID=$pid PHYSICAL_GPU=1 UUID=$uuid LOG=$log"
