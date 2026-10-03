#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$PROJECT_DIR/cache/pip" "$PROJECT_DIR/cache/tmp" "$PROJECT_DIR/cache/jupyter"
export PIP_CACHE_DIR="$PROJECT_DIR/cache/pip"
export TMPDIR="$PROJECT_DIR/cache/tmp"

if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
fi
TASK_PYTHON="$PROJECT_DIR/.venv/bin/python"
"$TASK_PYTHON" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
"$TASK_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$TASK_PYTHON" -m pip check
"$TASK_PYTHON" -m ipykernel install --prefix "$PROJECT_DIR/.venv" \
    --name math-reasoning --display-name 'Math reasoning'
"$TASK_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
