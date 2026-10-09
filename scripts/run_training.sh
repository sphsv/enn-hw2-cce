#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
implementation="${1:?usage: scripts/run_training.sh cce|torch_compile [official training arguments...]}"
shift

if [[ "$implementation" != "cce" && "$implementation" != "torch_compile" ]]; then
  echo "implementation must be cce or torch_compile" >&2
  exit 2
fi

raw_log="${RAW_LOG:-$project_root/results/raw/training_${implementation}.csv}"
official_repo="$project_root/external/ml-cross-entropy"
mkdir -p "$project_root/.cache/huggingface" "$project_root/checkpoints/$implementation"
export PYTHONPATH="$official_repo:$project_root"
export HF_HOME="$project_root/.cache/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
cd "$official_repo"

if [[ $# -eq 0 ]]; then
  echo "official training arguments are required; use scripts/run_figure4_gemma.sh" >&2
  exit 2
fi

exec "$project_root/.venv/bin/python" -m src.training_entrypoint \
  --implementation "$implementation" \
  --raw-log "$raw_log" \
  "$@"
