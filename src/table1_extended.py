from __future__ import annotations

import argparse
import gc
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from benchmark import data, memory
from benchmark.__main__ import baseline
from cut_cross_entropy import linear_cross_entropy

from .repro_utils import PROJECT_ROOT, load_config


LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float | None], torch.Tensor]


def clear_grad(embedding: torch.Tensor, classifier: torch.Tensor, *_args, **_kwargs) -> None:
    embedding.grad = None
    classifier.grad = None


def official_cce(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="cce",
    )


def cce_no_grad_filter(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="cce",
        filter_eps=None,
    )


def cce_kahan(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    # The pinned commit has no named "cce_kahan" preset. This uses current
    # FP32 accumulation flags; it does not recreate the paper's Kahan kernels.
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="cce",
        accum_e_fp32=True,
        accum_c_fp32=True,
    )


def cce_kahan_full_c(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="cce_kahan_full_c",
    )


def cce_kahan_full_e(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="cce_kahan_full_e",
    )


def torch_compile_loss(
    embedding: torch.Tensor,
    classifier: torch.Tensor,
    targets: torch.Tensor,
    softcap: float | None = None,
) -> torch.Tensor:
    return linear_cross_entropy(
        embedding,
        classifier,
        targets,
        softcap=softcap,
        impl="torch_compile",
    )


def build_methods(softcap: float | None) -> dict[str, LossFn]:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from torchtune.modules.loss import CEWithChunkedOutputLoss

    liger_loss = LigerFusedLinearCrossEntropyLoss(softcap=softcap)
    # Official TorchTune CEWithChunkedOutputLoss baseline from Table 1.
    torchtune_loss = CEWithChunkedOutputLoss(num_output_chunks=8)
    # This is the execution path recommended by TorchTune: compile only the
    # per-chunk upcast + CE, not the Python loop or all eight output projections.
    torchtune_loss.compute_cross_entropy = torch.compile(
        torchtune_loss.compute_cross_entropy,
        fullgraph=True,
    )

    def liger(
        embedding: torch.Tensor,
        classifier: torch.Tensor,
        targets: torch.Tensor,
        softcap: float | None = None,
    ) -> torch.Tensor:
        del softcap
        return liger_loss(classifier, embedding, targets)

    def torchtune_8_chunks(
        embedding: torch.Tensor,
        classifier: torch.Tensor,
        targets: torch.Tensor,
        softcap: float | None = None,
    ) -> torch.Tensor:
        # TorchTune expects [B, T, V] output chunks and [B, T] labels.  The
        # benchmark tensors are flattened [B*T, D]/[B*T], so use a singleton
        # batch dimension and split the token dimension into eight equal parts.
        logits_chunks = []
        for embedding_chunk in embedding.unsqueeze(0).chunk(8, dim=1):
            logits = embedding_chunk @ classifier.T
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            logits_chunks.append(logits)
        return torchtune_loss(logits_chunks, targets.unsqueeze(0))

    return {
        "cce": official_cce,
        "liger": liger,
        "torchtune_8_chunks": torchtune_8_chunks,
        "torch_compile": torch_compile_loss,
        "baseline": baseline,
        "cce_no_grad_filter": cce_no_grad_filter,
        "cce_kahan": cce_kahan,
        "cce_kahan_full_c": cce_kahan_full_c,
        "cce_kahan_full_e": cce_kahan_full_e,
    }


def timing_semantics(method: str, kind: str) -> str:
    if method == "liger" and kind == "loss-fw":
        return "includes_gradient_precomputation"
    if method == "liger" and kind == "loss-bw":
        return "gradient_compute_is_in_forward; backward_returns_saved_gradients"
    if kind == "loss-fw-bw":
        return "end_to_end_loss_and_gradient"
    return "standard_phase_timing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "raw" / "table1_extended_reps.csv"),
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = config["table1_extended"]
    iterations = args.iterations or int(experiment["iterations"])
    repetitions = args.repetitions or int(experiment["repetitions"])
    dtype = getattr(torch, config["dtype"])
    generator = data.generator(str(experiment["test_data"]))
    rows: list[dict[str, object]] = []

    for repeat in range(repetitions + 1):
        sample = generator(dtype=dtype)
        methods = build_methods(sample.softcap)
        for kind in experiment["kinds"]:
            embedding = sample.embedding.requires_grad_(True)
            classifier = sample.classifier.requires_grad_(True)
            targets = sample.targets
            kwargs: dict[str, object] = {"softcap": sample.softcap}
            if kind in {"loss-bw", "loss-fw-bw"}:
                kwargs.update(
                    backward=True,
                    forward=kind == "loss-fw-bw",
                    pre_fn=clear_grad,
                )

            for method_name in experiment["methods"]:
                status, error = "ok", ""
                try:
                    stats = memory.Stats.measure(
                        methods[method_name],
                        embedding,
                        classifier,
                        targets,
                        n_iteration=iterations if repeat > 0 else 1,
                        **kwargs,
                    )
                except Exception as exc:
                    status, error = "error", f"{type(exc).__name__}: {exc}"[:500]
                    stats = memory.Stats(float("nan"), float("nan"))

                if repeat > 0:
                    rows.append(
                        {
                            "method": method_name,
                            "kind": kind,
                            **asdict(stats),
                            "test_data": experiment["test_data"],
                            "repeat": repeat,
                            "n_iterations": iterations,
                            "dtype": config["dtype"],
                            "seed": config["seed"],
                            "gpu": torch.cuda.get_device_name(0),
                            "official_commit": config["official_commit"],
                            "vocab_size": sample.classifier.size(0),
                            "num_tokens": sample.embedding.size(0),
                            "hidden_dim": sample.embedding.size(1),
                            "softcap": sample.softcap,
                            "input_source": "gemma2_weights_alpaca_hidden_states",
                            "timing_semantics": timing_semantics(method_name, kind),
                            "status": status,
                            "error": error,
                        }
                    )

                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    failures = sum(row["status"] != "ok" for row in rows)
    print(f"Wrote {len(rows)} rows to {output}; failures={failures}")


if __name__ == "__main__":
    main()
