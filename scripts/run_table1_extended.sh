#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_repo="$project_root/external/ml-cross-entropy"
export PYTHONPATH="$official_repo:$project_root"
export HF_HOME="$project_root/.cache/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
cd "$project_root"
exec .venv/bin/python -m src.table1_extended "$@"
