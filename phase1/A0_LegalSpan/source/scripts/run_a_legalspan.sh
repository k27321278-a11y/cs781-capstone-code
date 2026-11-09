#!/usr/bin/env bash
set -euo pipefail
cd /home/stud1/cs781_a_legalspan_v1
TASK=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_legalspan_v1
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:.
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
test -s "$TASK/prepared_v2/manifest.json"
test -s "$TASK/gpu_training_verification.json"
test -s "$TASK/smoke_v2/smoke_evidence.json"
test -s "$TASK/public_context_preflight.json"
test -s docs/FINAL_REVIEW.md
test -s docs/REVIEW_RESOLUTION.md
# Take an exclusive launch lock before any preflight or handle write.
exec 9>"$TASK/launch.lock"
flock -n 9 || { echo 'An A supervisor already holds the launch lock'; exit 2; }
"$PY" - <<'PY'
import json, pathlib, subprocess
protected=[284634,469837,469841,497086,524833,526131,526372]
for pid in protected:
    state=subprocess.check_output(['ps','-p',str(pid),'-o','stat='],text=True).strip()
    if not state.startswith('T'): raise RuntimeError(f'Protected process {pid} changed state: {state}')
for proc in pathlib.Path('/proc').glob('[0-9]*'):
    try: args=(proc/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError): continue
    if b'legalspan.train' in args or b'legalspan.pipeline' in args:
        raise RuntimeError(f'Existing A process {proc.name}; refusing duplicate')
free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free < 8000: raise RuntimeError(f'GPU1 free memory changed: {free} MiB')
print(json.dumps({'gpu1_free_mib':free,'protected_processes_stopped':protected}),flush=True)
PY
TEST_PATH="$TASK/official_test.json"
test -s "$TEST_PATH"
nohup "$PY" -u -m legalspan.pipeline --data "$TASK/prepared_v2" --run "$TASK/experiment" --scorer-cache /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/bert_score_hf_cache --test "$TEST_PATH" --config configs/a_legalspan.json >"$TASK/supervisor.log" 2>&1 </dev/null &
TASKPID=$!
echo "$TASKPID" > "$TASK/supervisor.pid"
ps -ww -p "$TASKPID" -o pid,stat,etime,args
# Child inherits descriptor9 and holds launch.lock across every training/export.
