#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

# Keep caches/temp files inside this project directory.
mkdir -p .cache/huggingface .cache/torch .tmp
cat > .env_runtime <<ENV
export HF_HOME="$ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$ROOT/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"
export TMPDIR="$ROOT/.tmp"
ENV

echo "Environment ready. Activate with:"
echo "  source $ROOT/.venv/bin/activate"
echo "  source $ROOT/.env_runtime"
