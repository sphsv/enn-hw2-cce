#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_repo="$project_root/external/ml-cross-entropy"
methods="${METHODS:-cce,torch_compile,baseline}"
kinds="${KINDS:-loss-fw,loss-bw,loss-fw-bw}"
iterations="${ITERATIONS:-50}"
repetitions="${REPETITIONS:-3}"
output="${OUTPUT:-$project_root/results/raw/official_benchmark_reps.csv}"

mkdir -p "$(dirname "$output")" "$project_root/.cache/huggingface"
cd "$official_repo"
export PYTHONPATH="$official_repo:$project_root"
export HF_HOME="$project_root/.cache/huggingface"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"

"$project_root/.venv/bin/python" -m benchmark \
  --methods="$methods" \
  --test_data=gemma2 \
  --kinds="$kinds" \
  --n_iteration="$iterations" \
  --n_rep="$repetitions" \
  --dtype=bfloat16 \
  --output="$output"

cd "$project_root"
exec .venv/bin/python -m src.annotate_official \
  --output "$output" \
  --test-data gemma2 \
  --iterations "$iterations" \
  --repetitions "$repetitions"
