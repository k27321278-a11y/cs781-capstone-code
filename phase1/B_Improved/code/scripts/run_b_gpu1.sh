#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/stud1/cs781_phase1_v2
PY=/home/stud1/cs781_phase1_a_upload/CS781_Phase1_A_RTX_Server/.venv/bin/python
DATA=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2/official_100m_chars_per_source/prepared
ARTIFACT_ROOT=/media/hdd/SAMBA4080_left_Dataset/stud1/cs781_phase1_v2
BENCH_DIR="$ARTIFACT_ROOT/b_benchmark_gpu1_20261003"
RUN_DIR="$ARTIFACT_ROOT/b_phase1_5pass_20261003"
BENCH_LOG="$ARTIFACT_ROOT/b_benchmark_gpu1_20261003.log"
TRAIN_LOG="$ARTIFACT_ROOT/b_phase1_5pass_20261003.log"

fail() { printf 'B launch stopped: %s\n' "$1" >&2; exit 2; }

cd "$PROJECT"
[[ -x "$PY" ]] || fail "expected Python environment is unavailable: $PY"
[[ -f cs781/cli_b.py && -f configs/b.json ]] || fail "B bundle files are missing from $PROJECT"
[[ -f "$DATA/manifest.json" ]] || fail "prepared official data is missing: $DATA"
mountpoint -q /media/hdd || fail "/media/hdd is not mounted; refusing to write artifacts elsewhere"
[[ -d "$ARTIFACT_ROOT" ]] || fail "the designated HDD artifact directory is missing: $ARTIFACT_ROOT"

exec 9>"$ARTIFACT_ROOT/.b_gpu1_launch.lock"
flock -n 9 || fail "another B launch supervisor holds the lock"
if pgrep -u "$(id -u)" -f '[c]s781.cli_b'; then
    fail "a B command is already running; inspect it before launching another"
fi

free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ ]] || fail "could not read GPU1 free memory"
(( free_mib >= 8192 )) || fail "GPU1 has only ${free_mib} MiB free; B needs at least 8192 MiB for the measured run"

if pgrep -u "$(id -u)" -f '[c]s781.cli train-a'; then
    printf 'A training process is visible; leaving it untouched.\n'
else
    printf 'No matching active A CLI process was visible at launch time.\n'
fi
printf 'GPU1 initial free memory: %s MiB\n' "$free_mib"
nvidia-smi -i 1

[[ ! -e "$BENCH_DIR" && ! -e "$BENCH_LOG" ]] || fail "benchmark path already exists; preserve it and choose a fresh run name"
[[ ! -e "$RUN_DIR/checkpoints/latest.pt" ]] || fail "B checkpoint already exists; inspect it and explicitly resume instead"

export PYTHONPATH=.runtime:.test-deps:.
export CUDA_VISIBLE_DEVICES=1

printf 'Running B architecture/cache/masking tests before the GPU benchmark.\n'
"$PY" -m pytest tests/test_arch_b.py -q

printf 'Starting the 125-step B GPU1 sizing benchmark.\n'
"$PY" -u -m cs781.cli_b benchmark \
    --data "$DATA" \
    --run-dir "$BENCH_DIR" \
    --target-tokens 209914685 \
    --config configs/b.json \
    --device cuda >"$BENCH_LOG" 2>&1

[[ -s "$BENCH_DIR/benchmark.json" ]] || fail "B benchmark did not write benchmark.json; inspect $BENCH_LOG"
free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 8192 )) || fail "GPU1 availability changed after benchmarking; no full run launched"
[[ ! -e "$RUN_DIR/checkpoints/latest.pt" ]] || fail "B checkpoint appeared before launch; refusing to overwrite it"

printf 'Benchmark passed; starting full B run for 209914685 supervised target tokens on visible GPU1.\n'
printf 'Training output is redirected to %s\n' "$TRAIN_LOG"
"$PY" -u -m cs781.cli_b train \
    --data "$DATA" \
    --run-dir "$RUN_DIR" \
    --target-tokens 209914685 \
    --config configs/b.json \
    --device cuda >"$TRAIN_LOG" 2>&1

printf 'B training process exited. Inspect training_status.json, the final metrics row, and the checkpoint before evaluation.\n'
