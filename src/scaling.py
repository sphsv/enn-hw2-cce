from __future__ import annotations

import argparse
import csv
import gc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cut_cross_entropy import linear_cross_entropy

from .repro_utils import PROJECT_ROOT, baseline_cross_entropy, load_config, set_seed, torch_dtype


LossFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def method_loss(method: str) -> LossFunction:
    if method == "baseline":
        return baseline_cross_entropy
    return lambda e, c, targets: linear_cross_entropy(e, c, targets, impl=method)


def _clear() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _execute_once(
    base_e: torch.Tensor,
    base_c: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFunction,
    operation: str,
) -> tuple[float, float]:
    e = base_e.detach().clone().requires_grad_(operation != "forward")
    c = base_c.detach().clone().requires_grad_(operation != "forward")

    loss: torch.Tensor | None = None
    if operation == "backward":
        loss = loss_fn(e, c, targets)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    if operation == "forward":
        loss = loss_fn(e, c, targets)
    elif operation == "backward":
        assert loss is not None
        loss.backward()
    elif operation == "forward_backward":
        loss = loss_fn(e, c, targets)
        loss.backward()
    else:
        raise ValueError(f"Unknown operation: {operation}")

    end.record()
    torch.cuda.synchronize()
    runtime_ms = start.elapsed_time(end)
    peak_delta_mb = (torch.cuda.max_memory_allocated() - allocated_before) / 2**20

    del loss, e, c
    return runtime_ms, peak_delta_mb


def measure(
    base_e: torch.Tensor,
    base_c: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: LossFunction,
    operation: str,
    warmups: int,
    repetitions: int,
) -> list[tuple[float, float]]:
    for _ in range(warmups):
        _execute_once(base_e, base_c, targets, loss_fn, operation)
    _clear()
    return [
        _execute_once(base_e, base_c, targets, loss_fn, operation)
        for _ in range(repetitions)
    ]


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    scaling: dict[str, Any] = config["scaling"].copy()

    dtype = torch_dtype(config["dtype"])
    seed = int(scaling["seed"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "operation",
        "vocab_size",
        "num_tokens",
        "hidden_dim",
        "dtype",
        "runtime_ms",
        "peak_memory_mb",
        "repeat",
        "gpu",
        "seed",
        "official_commit",
        "status",
        "error",
    ]

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vocab_size in scaling["vocab_sizes"]:
            set_seed(seed)
            m, d = int(scaling["num_tokens"]), int(scaling["hidden_dim"])
            scale = d**-0.25
            base_e = torch.randn(m, d, device="cuda", dtype=dtype) * scale
            base_c = torch.randn(int(vocab_size), d, device="cuda", dtype=dtype) * scale
            targets = torch.randint(int(vocab_size), (m,), device="cuda")
            for operation in scaling["operations"]:
                for method in scaling["methods"]:
                    common = {
                        "method": method,
                        "operation": operation,
                        "vocab_size": vocab_size,
                        "num_tokens": m,
                        "hidden_dim": d,
                        "dtype": config["dtype"],
                        "gpu": torch.cuda.get_device_name(0),
                        "seed": seed,
                        "official_commit": config["official_commit"],
                    }
                    try:
                        measurements = measure(
                            base_e,
                            base_c,
                            targets,
                            method_loss(method),
                            operation,
                            int(scaling["warmups"]),
                            int(scaling["repetitions"]),
                        )
                        for repeat, (runtime_ms, peak_memory_mb) in enumerate(measurements):
                            writer.writerow(
                                common
                                | {
                                    "runtime_ms": runtime_ms,
                                    "peak_memory_mb": peak_memory_mb,
                                    "repeat": repeat,
                                    "status": "ok",
                                    "error": "",
                                }
                            )
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        writer.writerow(
                            common
                            | {
                                "runtime_ms": "",
                                "peak_memory_mb": "",
                                "repeat": "",
                                "status": "error",
                                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                            }
                        )
                        _clear()
                    handle.flush()
            del base_e, base_c, targets
            _clear()
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "raw" / "scaling.csv"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
