from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .repro_utils import PROJECT_ROOT, load_config, write_json


def command(*args: str) -> str | None:
    try:
        return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "raw" / "environment.json"))
    parser.add_argument(
        "--official-repo",
        default=str(PROJECT_ROOT / "external" / "ml-cross-entropy"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    repo = Path(args.official_repo)
    commit = command("git", "-C", str(repo), "rev-parse", "HEAD")
    commit_date = command("git", "-C", str(repo), "show", "-s", "--format=%cI", "HEAD")
    tracked_changes = command(
        "git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"
    )
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "triton",
                "transformers",
                "cut-cross-entropy",
                "liger-kernel",
                "torchtune",
                "torchao",
                "numpy",
                "pandas",
                "matplotlib",
                "datasets",
                "accelerate",
                "deepspeed",
                "huggingface-hub",
                "tokenizers",
                "fire",
                "tqdm",
                "pytest",
            )
        },
        "gpu_query": command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ),
        "nvidia_smi_header": command("nvidia-smi"),
        "official_repo": str(repo.resolve()),
        "official_commit": commit,
        "official_commit_date": commit_date,
        "official_tracked_tree_clean": tracked_changes == "",
        "expected_official_commit": config["official_commit"],
        "commit_matches_config": commit == config["official_commit"],
        "seed": config["seed"],
        "config": config,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
