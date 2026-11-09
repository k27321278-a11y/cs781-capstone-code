#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/stud1/cs781_b_ownbpe_v1
ART=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/own_bpe_20261006
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
PARENT="$ART/b0_scratch"
IMPROVED="$ART/b1_improved"
DATA="$ART/prepared"
ALIGNED="$ART/b_taskaligned"
OUTPUT="$ART/CS781_B1_OwnBPE_Codabench.zip"
SCORER=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/bert_score_hf_cache

export PYTHONPATH=/home/stud1/cs781_phase1_v2/.runtime:/home/stud1/cs781_phase1_v2/.test-deps:$ROOT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
test -s "$DATA/manifest.json"
test -s "$ART/official_test.json"
test -d "$SCORER"
test ! -e "$OUTPUT"

echo "B1 stage 1: scratch encoder-decoder pretraining with project-owned BPE on physical GPU0"
parent_resume=()
if [[ -f "$PARENT/checkpoints/latest.pt" ]]; then parent_resume=(--resume); fi
"$PY" -u -m cs781.cli_b train --data "$DATA" --run-dir "$PARENT" \
  --config configs/b_ownbpe.json --device cuda "${parent_resume[@]}"

echo "B1 stage 2: task-aligned continuation, dev selection, and official ZIP export"
improved_resume=()
if [[ -f "$IMPROVED/checkpoints/latest.pt" ]]; then improved_resume=(--resume); fi
"$PY" -u scripts/improve_b.py \
  --parent-checkpoint "$PARENT/checkpoints/latest.pt" \
  --data "$DATA" --catalog "$ART/catalog.sqlite" \
  --run-dir "$IMPROVED" --aligned-data "$ALIGNED" \
  --scorer-cache "$SCORER" --output "$OUTPUT" \
  --config configs/b_improved_ownbpe.json --test-file "$ART/official_test.json" \
  "${improved_resume[@]}"
test -s "$OUTPUT"
sha256sum "$OUTPUT"
