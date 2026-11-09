#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
[[ -f .env_runtime ]] && source .env_runtime

WORKDIR="${WORKDIR:-$ROOT/runs/A0_dense_flash}"
mkdir -p "$WORKDIR/logs"

python architecture_a_legalgpt.py --mode selftest

python architecture_a_legalgpt.py --mode prepare \
  --workdir "$WORKDIR" \
  --vocab_size 32768 \
  --tokenizer_train_chars 100000000 \
  --prepare_train_tokens 200000000 \
  --prepare_val_tokens 5000000 \
  --val_generation_examples 128 \
  --mix_mode equal 2>&1 | tee "$WORKDIR/logs/prepare.log"
