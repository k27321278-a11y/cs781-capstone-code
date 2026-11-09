#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$ROOT/runs/A0_dense_flash}"
echo "=== GPUs ==="
nvidia-smi
echo
echo "=== Last 40 training log lines ==="
tail -n 40 "$WORKDIR/logs/train.log" 2>/dev/null || echo "train.log not created yet"
