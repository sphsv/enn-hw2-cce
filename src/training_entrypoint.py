from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers


class RawMetricsCallback(transformers.TrainerCallback):
    def __init__(self, output: Path):
        self.output = output
        self.started_at = 0.0
        self.tokens = 0
        self.pending: dict[int, dict[str, Any]] = {}
        self.rows: list[dict[str, Any]] = []

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started_at = time.perf_counter()
        self.tokens = 0

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started_at
        self.pending[int(state.global_step)] = {
            "step": int(state.global_step),
            "step_time": elapsed,
            "tokens": self.tokens,
            "tokens_per_second": self.tokens / elapsed if elapsed else 0.0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
            "allocated_gpu_memory_mb": torch.cuda.memory_allocated() / 2**20,
            "reserved_gpu_memory_mb": torch.cuda.memory_reserved() / 2**20,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        step = int(state.global_step)
        row = self.pending.pop(step, {"step": step})
        logs = logs or {}
        row.update(
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            epoch=logs.get("epoch"),
            eval_loss=logs.get("eval_loss"),
            perplexity=logs.get("eval_perplexity"),
        )
        self.rows.append(row)
        self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "step",
            "train_loss",
            "learning_rate",
            "step_time",
            "tokens",
            "tokens_per_second",
            "peak_gpu_memory_mb",
            "allocated_gpu_memory_mb",
            "reserved_gpu_memory_mb",
            "eval_loss",
            "perplexity",
            "epoch",
        ]
        with self.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in self.rows)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--implementation",
        choices=("cce", "torch_compile"),
        required=True,
    )
    parser.add_argument("--raw-log", required=True)
    parser.add_argument("--manifest")
    known, forwarded = parser.parse_known_args()

    started_at = datetime.now(timezone.utc)
    manifest_path = (
        Path(known.manifest)
        if known.manifest
        else Path(known.raw_log).with_name(
            f"{Path(known.raw_log).stem}_manifest.json"
        )
    )
    status = "failed"

    try:
        import training.train as official_train

        callback = RawMetricsCallback(Path(known.raw_log))
        base_trainer = official_train.transformers.Trainer

        # Reuse the upstream training entrypoint unchanged. Only Trainer is
        # instrumented to record per-step metrics; model and training logic
        # remain the upstream implementation.
        class ReproTrainer(base_trainer):
            def __init__(self, *args, **kwargs):
                callbacks = list(kwargs.pop("callbacks", []) or [])
                callbacks.append(callback)
                super().__init__(*args, callbacks=callbacks, **kwargs)

            def training_step(self, model, inputs, *args, **kwargs):
                labels = inputs.get("labels")
                if labels is not None:
                    callback.tokens += int((labels != -100).count_nonzero())
                return super().training_step(model, inputs, *args, **kwargs)

        official_train.transformers.Trainer = ReproTrainer
        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        sys.argv = [sys.argv[0], *forwarded]
        official_train.main()
        status = "complete"
    finally:
        completed_at = datetime.now(timezone.utc)
        manifest = {
            "implementation": known.implementation,
            "status": status,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "wall_time_seconds": (completed_at - started_at).total_seconds(),
            "raw_log": str(Path(known.raw_log)),
            "forwarded_arguments": forwarded,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
