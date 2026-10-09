from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from cut_cross_entropy import linear_cross_entropy

from .repro_utils import (
    PROJECT_ROOT,
    baseline_cross_entropy,
    load_config,
    scalar_error,
    set_seed,
    tensor_error_metrics,
    torch_dtype,
    write_json,
)


def evaluate_method(
    method: str,
    base_e: torch.Tensor,
    base_c: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    e = base_e.detach().clone().requires_grad_(True)
    c = base_c.detach().clone().requires_grad_(True)

    if method == "baseline":
        loss = baseline_cross_entropy(e, c, targets)
    else:
        loss = linear_cross_entropy(e, c, targets, impl=method)
    loss.backward()
    assert e.grad is not None and c.grad is not None
    return {
        "loss": float(loss.detach()),
        "embedding_grad": e.grad.detach().clone(),
        "classifier_grad": c.grad.detach().clone(),
    }


def run_case(case: dict[str, int], dtype: torch.dtype, atol: float, rtol: float) -> dict[str, Any]:
    seed = case["seed"]
    set_seed(seed)
    device = torch.device("cuda")
    m, v, d = case["num_tokens"], case["vocab_size"], case["hidden_dim"]
    scale = d**-0.25
    base_e = torch.randn(m, d, device=device, dtype=dtype) * scale
    base_c = torch.randn(v, d, device=device, dtype=dtype) * scale
    targets = torch.randint(v, (m,), device=device)

    raw = {
        method: evaluate_method(method, base_e, base_c, targets)
        for method in ("baseline", "torch_compile", "cce_exact", "cce")
    }
    reference = raw["baseline"]
    comparisons: dict[str, Any] = {}
    for method in ("torch_compile", "cce_exact", "cce"):
        result = raw[method]
        comparisons[method] = {
            "loss": scalar_error(reference["loss"], result["loss"]),
            "embedding_gradient": tensor_error_metrics(
                reference["embedding_grad"], result["embedding_grad"], atol=atol, rtol=rtol
            ),
            "classifier_gradient": tensor_error_metrics(
                reference["classifier_grad"], result["classifier_grad"], atol=atol, rtol=rtol
            ),
        }

    return {
        **case,
        "dtype": str(dtype).removeprefix("torch."),
        "losses": {method: result["loss"] for method, result in raw.items()},
        "comparisons_to_baseline": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "raw" / "correctness.json"))
    args = parser.parse_args()

    config = load_config(args.config)
    correctness = config["correctness"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": config["official_commit"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cases": [
            run_case(
                case,
                torch_dtype(config["dtype"]),
                float(correctness["atol"]),
                float(correctness["rtol"]),
            )
            for case in correctness["cases"]
        ],
    }
    write_json(Path(args.output), payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
