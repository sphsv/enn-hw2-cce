from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .repro_utils import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output)
    frame = pd.read_csv(output)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
    frame["repeat"] = frame.groupby(["method", "kind", "test_data"]).cumcount() + 1
    frame["n_iterations"] = args.iterations
    frame["dtype"] = config["dtype"]
    frame["seed"] = config["seed"]
    frame["gpu"] = torch.cuda.get_device_name(0)
    frame["official_commit"] = config["official_commit"]

    if args.test_data != "gemma2":
        raise ValueError("Only the exact Gemma-2 benchmark is part of this reproduction")
    shape = config["official_benchmark"]
    frame["vocab_size"] = shape["vocab_size"]
    frame["num_tokens"] = shape["num_tokens"]
    frame["hidden_dim"] = shape["hidden_dim"]
    frame["softcap"] = shape["softcap"]
    frame["input_source"] = "gemma2_weights_alpaca_hidden_states"

    if len(frame) != args.repetitions * frame[["method", "kind"]].drop_duplicates().shape[0]:
        raise RuntimeError("Unexpected official benchmark row count")
    frame.to_csv(output, index=False)
    print(f"Annotated {output}")


if __name__ == "__main__":
    main()
