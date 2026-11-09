#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
[[ -f .env_runtime ]] && source .env_runtime
source ./select_gpu.sh
WORKDIR="${WORKDIR:-$ROOT/runs/A0_dense_flash}"

python architecture_a_legalgpt.py --mode submit \
  --workdir "$WORKDIR" \
  --strategy greedy

echo
echo "UPLOAD THIS FILE TO CODABENCH:"
echo "$WORKDIR/submission/greedy/submission.zip"
unzip -l "$WORKDIR/submission/greedy/submission.zip"
