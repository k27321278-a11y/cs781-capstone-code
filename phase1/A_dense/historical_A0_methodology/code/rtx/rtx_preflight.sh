#!/usr/bin/env bash
set -euo pipefail

echo "=== CS781 Architecture A RTX preflight ==="
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }

echo
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo
echo "Active compute processes (if any):"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo
echo "Disk:"
df -h "${PWD}"

echo
echo "Python:"
python3 --version || true

echo
echo "Tip: if FeelWM is using one GPU, train CS781 only on another sufficiently free GPU."
