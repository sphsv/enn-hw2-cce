from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    config = json.loads((ROOT / "configs" / "experiments.json").read_text())
    environment = json.loads((RAW / "environment.json").read_text())
    assert environment["official_commit"] == config["official_commit"]
    assert environment["commit_matches_config"] is True
    assert environment["official_tracked_tree_clean"] is True
    # Keep the historical run configuration intact. The active configuration
    # may exclude methods after review, but may not alter the other settings.
    captured_config = deepcopy(environment["config"])
    active_methods = config["table1_extended"]["methods"]
    assert set(active_methods) <= set(captured_config["table1_extended"]["methods"])
    captured_config["table1_extended"]["methods"] = active_methods
    assert captured_config == config

    table_config = config["table1_extended"]
    table1 = rows(RAW / "table1_extended_reps.csv")
    assert len(table1) == (
        len(table_config["methods"])
        * len(table_config["kinds"])
        * table_config["repetitions"]
    )
    assert all(row["status"] == "ok" for row in table1)
    expected_grid = set(product(
        table_config["methods"], table_config["kinds"],
        range(1, table_config["repetitions"] + 1),
    ))
    measured_grid = [
        (row["method"], row["kind"], int(row["repeat"])) for row in table1
    ]
    assert len(set(measured_grid)) == len(measured_grid)
    assert set(measured_grid) == expected_grid
    assert all(
        math.isfinite(float(row[field])) and float(row[field]) >= 0
        for row in table1 for field in ("runtime_ms", "op_mem_mb")
    )
    assert all(
        row["timing_semantics"]
        == "gradient_compute_is_in_forward; backward_returns_saved_gradients"
        for row in table1
        if row["method"] == "liger" and row["kind"] == "loss-bw"
    )

    scaling_config = config["scaling"]
    scaling = rows(RAW / "scaling.csv")
    assert len(scaling) == (
        len(scaling_config["vocab_sizes"])
        * len(scaling_config["methods"])
        * len(scaling_config["operations"])
        * scaling_config["repetitions"]
    )
    assert all(row["status"] == "ok" for row in scaling)

    experiment = config["figure4_gemma_training"]
    steps = experiment["max_steps"]
    training_runs = {}
    for implementation in experiment["implementations"]:
        for seed in experiment["seeds"]:
            stem = f"training_figure4_gemma_{implementation}_seed{seed}"
            run = [row for row in rows(RAW / f"{stem}.csv") if row["train_loss"]]
            training_runs[implementation, seed] = run
            assert [int(row["step"]) for row in run] == list(range(1, steps + 1))
            manifest = json.loads((RAW / f"{stem}_manifest.json").read_text())
            assert manifest["status"] == "complete"
            assert manifest["implementation"] == implementation

    for seed in experiment["seeds"]:
        cce = training_runs["cce", seed]
        control = training_runs["torch_compile", seed]
        assert [row["tokens"] for row in cce] == [row["tokens"] for row in control]

    result = {
        "status": "ok",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "table1_rows": len(table1),
        "scaling_rows": len(scaling),
        "figure4_runs": len(training_runs),
        "figure4_steps_per_run": steps,
    }
    (PROCESSED / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
