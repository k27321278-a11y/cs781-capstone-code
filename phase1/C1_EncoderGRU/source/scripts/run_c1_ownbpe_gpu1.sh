#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/stud1/cs781_c1_ownbpe_v1
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
RUN="$ART/c1_scratch"
DATA="$ART/prepared"
OUTPUT="$ART/CS781_C1_OwnBPE_Codabench.zip"

export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:$ROOT
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
test -s "$DATA/manifest.json"
test -s "$ART/official_test.json"
test ! -e "$OUTPUT"

echo "C1 stage 1: scratch Transformer-GRU pretraining with project-owned BPE on physical GPU1"
resume=()
if [[ -f "$RUN/checkpoints/latest.pt" ]]; then resume=(--resume); fi
"$PY" -u -m cs781.cli_c train --data "$DATA" --run-dir "$RUN" \
  --config configs/c1_ownbpe.json --device cuda "${resume[@]}"

echo "C1 stage 2: pinned official test export with the original C policy-2 decoder"
"$PY" -u -m cs781.cli_c predict-official-test \
  --data "$DATA" --checkpoint "$RUN/checkpoints/latest.pt" \
  --output "$OUTPUT" --config configs/c1_ownbpe.json --policy 2 \
  --test-file "$ART/official_test.json" --device cuda
test -s "$OUTPUT"
sha256sum "$OUTPUT"
