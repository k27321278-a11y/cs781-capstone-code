#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/stud1/cs781_b_ownbpe_v1
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:$ROOT
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false

cd "$ROOT"
test -L "$ART/catalog.sqlite"
test -s "$ART/tokenizer.json"
test -s "$ART/tokenizer.meta.json"
test ! -e "$ART/prepared"
exec "$PY" -u -m cs781.cli prepare \
  --root "$ART" \
  --revision 89edba6a44d6bb9593fd13b6895b5cd3b06eb82e \
  --max-chars-per-source 100000000 --tokenizer-chars 100000000 \
  --context 2048
