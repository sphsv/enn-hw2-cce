#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base_python="${BASE_PYTHON:-}"

if [[ -z "$base_python" ]]; then
  echo "Set BASE_PYTHON to a Python containing compatible torch and triton." >&2
  exit 1
fi

if [[ ! -x "$base_python" ]]; then
  echo "Existing GPU stack not found at $base_python" >&2
  echo "Set BASE_PYTHON to a Python containing compatible torch and triton." >&2
  exit 1
fi

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  python3 -m venv "$project_root/.venv"
fi

project_site="$($project_root/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
base_site="$($base_python -c 'import site; print(site.getsitepackages()[0])')"

"$project_root/.venv/bin/python" - "$project_site" "$base_site" <<'PY'
from pathlib import Path
import sys

project_site, base_site = map(Path, sys.argv[1:])
project_site.mkdir(parents=True, exist_ok=True)
(project_site / "shared_gpu_stack.pth").write_text(str(base_site.resolve()) + "\n")
PY

"$project_root/.venv/bin/pip" install -r "$project_root/requirements-benchmark.txt"
"$project_root/.venv/bin/pip" install --no-deps -e "$project_root/external/ml-cross-entropy"

if [[ "${1:-}" == "--training" ]]; then
  "$project_root/.venv/bin/pip" install -r "$project_root/requirements-training.txt"
fi

"$project_root/.venv/bin/python" -c \
  'import torch, triton, cut_cross_entropy; print(f"torch={torch.__version__} triton={triton.__version__} gpu={torch.cuda.get_device_name(0)}")'
