#!/usr/bin/env bash
set -euo pipefail

ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python

date -Is
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,ppid,stat,etime,args | grep -E '[c]s781.cli_b train|[c]s781.cli_c train|[i]mprove_b.py|[r]un_b1_ownbpe_gpu0|[r]un_c1_ownbpe_gpu1' || true

"$PY" - "$ART" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ('b0_scratch', 'b1_improved', 'c1_scratch'):
    run = root / name
    metadata = run / 'run.json'
    metrics = run / 'metrics.jsonl'
    status = run / 'training_status.json'
    if not metadata.exists():
        print(f'{name}: not started')
        continue
    target = json.loads(metadata.read_text())['train_config']['target_tokens']
    last = None
    if metrics.exists():
        with metrics.open('rb') as handle:
            lines = handle.readlines()
        if lines:
            last = json.loads(lines[-1])
    saved = json.loads(status.read_text()) if status.exists() else {}
    if last:
        tokens = last['target_tokens']
        rate = last.get('target_tokens_per_second') or 0
        eta = f'{max(0, target-tokens)/rate/3600:.2f} h' if rate > 0 else 'unknown'
        print(f'{name}: {saved.get("status", "unknown")} step={last["step"]} '
              f'target_tokens={tokens}/{target} rate={rate:.0f}/s ETA={eta} loss={last["loss"]:.3f}')
    else:
        print(f'{name}: {saved.get("status", "initializing")} target_tokens=0/{target}')

pipeline = root / 'b1_improved/pipeline_status.json'
if pipeline.exists():
    print('B1 pipeline:', pipeline.read_text().strip())
for name in ('CS781_B1_OwnBPE_Codabench.zip', 'CS781_C1_OwnBPE_Codabench.zip'):
    path = root / name
    print(f'{name}: {path.stat().st_size} bytes' if path.exists() else f'{name}: pending')
PY

tail -n 3 "$ART/b1_pipeline.log" 2>/dev/null || true
tail -n 3 "$ART/c1_pipeline.log" 2>/dev/null || true
