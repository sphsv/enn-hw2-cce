from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from cut_cross_entropy import linear_cross_entropy

from .repro_utils import (
    PROJECT_ROOT,
    load_config,
    set_seed,
    tensor_error_metrics,
    torch_dtype,
    write_json,
)
from .scaling import measure


def filtered_loss(e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return linear_cross_entropy(e, c, targets, impl="cce", filter_eps="auto")


def unfiltered_loss(e: torch.Tensor, c: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return linear_cross_entropy(
        e,
        c,
        targets,
        impl="cce",
        filter_eps=None,
        filter_e_grad=False,
        filter_c_grad=False,
    )


def gradients(
    fn, base_e: torch.Tensor, base_c: torch.Tensor, targets: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor]:
    e = base_e.detach().clone().requires_grad_(True)
    c = base_c.detach().clone().requires_grad_(True)
    loss = fn(e, c, targets)
    loss.backward()
    assert e.grad is not None and c.grad is not None
    return float(loss.detach()), e.grad.detach().clone(), c.grad.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "raw" / "filtering.csv"))
    parser.add_argument(
        "--correctness-output",
        default=str(PROJECT_ROOT / "results" / "raw" / "filtering_correctness.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    filtering: dict[str, Any] = config["filtering"].copy()
    dtype = torch_dtype(config["dtype"])
    rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []

    for seed in filtering["seeds"]:
        set_seed(int(seed))
        m = int(filtering["num_tokens"])
        v = int(filtering["vocab_size"])
        d = int(filtering["hidden_dim"])
        scale = d**-0.25
        base_e = torch.randn(m, d, device="cuda", dtype=dtype) * scale
        base_c = torch.randn(v, d, device="cuda", dtype=dtype) * scale
        targets = torch.randint(v, (m,), device="cuda")

        exact_loss, exact_e, exact_c = gradients(unfiltered_loss, base_e, base_c, targets)
        filt_loss, filt_e, filt_c = gradients(filtered_loss, base_e, base_c, targets)
        correctness_rows.append(
            {
                "seed": seed,
                "loss_unfiltered": exact_loss,
                "loss_filtered": filt_loss,
                "loss_absolute_error": abs(filt_loss - exact_loss),
                "embedding_gradient": tensor_error_metrics(
                    exact_e, filt_e, atol=0.005, rtol=0.05
                ),
                "classifier_gradient": tensor_error_metrics(
                    exact_c, filt_c, atol=0.005, rtol=0.05
                ),
            }
        )

        for name, fn in (("filtered_auto", filtered_loss), ("unfiltered", unfiltered_loss)):
            measurements = measure(
                base_e,
                base_c,
                targets,
                fn,
                "backward",
                int(filtering["warmups"]),
                int(filtering["repetitions"]),
            )
            for repeat, (runtime_ms, peak_memory_mb) in enumerate(measurements):
                rows.append(
                    {
                        "variant": name,
                        "seed": seed,
                        "num_tokens": m,
                        "vocab_size": v,
                        "hidden_dim": d,
                        "dtype": config["dtype"],
                        "runtime_ms": runtime_ms,
                        "peak_memory_mb": peak_memory_mb,
                        "repeat": repeat,
                        "gpu": torch.cuda.get_device_name(0),
                        "official_commit": config["official_commit"],
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.correctness_output, {"cases": correctness_rows})
    print(f"Wrote {output} and {args.correctness_output}")


if __name__ == "__main__":
    main()
