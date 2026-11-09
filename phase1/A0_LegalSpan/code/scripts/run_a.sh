#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 4 ]] || { echo 'Usage: run_a.sh DATA RUN_DIR PHYSICAL_GPU TARGET_TOKENS [--resume]' >&2; exit 2; }
CS781_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python}"
export PYTHONPATH="$CS781_ROOT/.runtime:$CS781_ROOT/.test-deps:$CS781_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$3"
"$PYTHON_BIN" -m cs781.cli validate-data --data "$1"
"$PYTHON_BIN" -u -m cs781.cli train-a --data "$1" --run-dir "$2" --target-tokens "$4" "${@:5}"
