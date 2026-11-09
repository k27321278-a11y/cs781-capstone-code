#!/usr/bin/env bash
set -euo pipefail
cd /home/stud1/cs781_a_legalspan_v1
TASK=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_legalspan_v1
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:.
if test -e "$TASK/tokenizer.json"; then echo 'Tokenizer already exists; refusing refit'; exit 3; fi
nohup flock -n "$TASK/tokenizer_fit.lock" "$PY" -u -m legalspan.bpe --catalog /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source/catalog.sqlite --output "$TASK/tokenizer.json" >"$TASK/tokenizer_fit.log" 2>&1 </dev/null &
TASKPID=$!
echo "$TASKPID" > "$TASK/tokenizer_fit.pid"
ps -ww -p "$TASKPID" -o pid,stat,etime,args
