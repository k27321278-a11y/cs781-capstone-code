#!/usr/bin/env bash
set -euo pipefail
cd /home/stud1/cs781_a_legalspan_v1
TASK=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/a_legalspan_v1
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:.
test -s "$TASK/tokenizer.meta.json"
if test -e "$TASK/prepared"; then echo 'Prepared data already exists'; exit 3; fi
nohup flock -n "$TASK/prepare.lock" "$PY" -u -m legalspan.data --catalog /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source/catalog.sqlite --parent-data /media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source/prepared --tokenizer "$TASK/tokenizer.json" --output "$TASK/prepared" >"$TASK/prepare.log" 2>&1 </dev/null &
TASKPID=$!
echo "$TASKPID" > "$TASK/prepare.pid"
ps -ww -p "$TASKPID" -o pid,stat,etime,args
