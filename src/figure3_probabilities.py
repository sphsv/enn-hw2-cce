from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from benchmark import data

from .repro_utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "raw" / "figure3_probabilities.csv"),
    )
    parser.add_argument("--chunk-size", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = config["figure3"]
    dtype = getattr(torch, config["dtype"])
    sample = data.generator(str(experiment["test_data"]))(dtype=dtype)
    chunk_size = args.chunk_size or int(experiment["chunk_size"])
    probability_sum = torch.zeros(sample.classifier.size(0), dtype=torch.float64)

    with torch.inference_mode():
        for start in range(0, sample.embedding.size(0), chunk_size):
            embedding = sample.embedding[start : start + chunk_size]
            logits = embedding @ sample.classifier.T
            if sample.softcap is not None:
                logits = torch.tanh(logits / sample.softcap) * sample.softcap
            probabilities = torch.softmax(logits.float(), dim=-1)
            ranked = probabilities.sort(dim=-1, descending=True).values
            probability_sum += ranked.sum(dim=0).double().cpu()

    average = probability_sum / sample.embedding.size(0)
    threshold = torch.finfo(dtype).eps / 32
    below = (average < threshold).nonzero(as_tuple=True)[0]
    crossing_rank = int(below[0]) + 1 if below.numel() else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "rank": range(1, average.numel() + 1),
            "average_probability": average.numpy(),
            "filter_threshold": threshold,
        }
    ).to_csv(output, index=False)
    metadata = {
        "test_data": experiment["test_data"],
        "num_tokens": sample.embedding.size(0),
        "vocab_size": sample.classifier.size(0),
        "hidden_dim": sample.embedding.size(1),
        "dtype": config["dtype"],
        "softcap": sample.softcap,
        "filter_threshold": threshold,
        "first_rank_below_threshold": crossing_rank,
        "gpu": torch.cuda.get_device_name(0),
        "official_commit": config["official_commit"],
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
