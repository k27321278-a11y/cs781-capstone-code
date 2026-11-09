#!/usr/bin/env bash
# Source this file:  source ./select_gpu.sh
# Optional: GPU=1 source ./select_gpu.sh
# Optional: MIN_FREE_MIB=9000 source ./select_gpu.sh
set -euo pipefail
MIN_FREE_MIB="${MIN_FREE_MIB:-9000}"

if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${GPU:-}" ]]; then
  chosen="$GPU"
else
  chosen=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2>m){m=$2;i=$1}} END{print i}')
fi

free_mib=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v idx="$chosen" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==idx) print $2}')
name=$(nvidia-smi --query-gpu=index,name --format=csv,noheader \
  | awk -F',' -v idx="$chosen" '{gsub(/^ +| +$/,"",$1); if ($1==idx){sub(/^ +/,"",$2); print $2}}')

if [[ -z "$free_mib" ]]; then
  echo "ERROR: GPU index $chosen not found" >&2
  return 1 2>/dev/null || exit 1
fi

echo "Selected physical GPU $chosen: $name, free=${free_mib} MiB"
if (( free_mib < MIN_FREE_MIB )); then
  echo "ERROR: only ${free_mib} MiB free; threshold is ${MIN_FREE_MIB} MiB." >&2
  echo "Do not compete with the existing FeelWM run on this GPU. Pick another GPU or wait." >&2
  return 2 2>/dev/null || exit 2
fi

export CUDA_VISIBLE_DEVICES="$chosen"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
