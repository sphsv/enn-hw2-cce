from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WARMUP_STEPS = 10
VALUE_FIELDS = (
    "train_loss",
    "step_time",
    "tokens",
    "tokens_per_second",
    "peak_gpu_memory_mb",
    "allocated_gpu_memory_mb",
    "reserved_gpu_memory_mb",
)
RUN_NAME = re.compile(r"training_figure4_gemma_(cce|torch_compile)_seed(\d+)\.csv$")


def read_training_rows(source: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with source.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if not raw.get("train_loss"):
                continue
            row: dict[str, float | int] = {"step": int(raw["step"])}
            row.update({field: float(raw[field]) for field in VALUE_FIELDS})
            rows.append(row)
    if [int(row["step"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Training steps in {source} are not consecutive")
    return rows


def run_summary(
    implementation: str,
    rows: list[dict[str, float | int]],
) -> dict[str, str | int | float]:
    steady = rows[WARMUP_STEPS:]
    losses = [float(row["train_loss"]) for row in rows]
    step_times = [float(row["step_time"]) for row in rows]
    steady_times = [float(row["step_time"]) for row in steady]
    return {
        "implementation": implementation,
        "steps": len(rows),
        "warmup_steps_excluded": WARMUP_STEPS,
        "loss_mean": statistics.fmean(losses),
        "loss_median": statistics.median(losses),
        "loss_last20_mean": statistics.fmean(losses[-20:]),
        "loss_last50_mean": statistics.fmean(losses[-50:]),
        "step_time_total_s": sum(step_times),
        "step_time_mean_s": statistics.fmean(step_times),
        "step_time_median_s": statistics.median(step_times),
        "tokens_per_second_aggregate": sum(float(row["tokens"]) for row in rows)
        / sum(step_times),
        "steady_step_time_mean_s": statistics.fmean(steady_times),
        "steady_step_time_median_s": statistics.median(steady_times),
        "steady_tokens_per_second_aggregate": sum(
            float(row["tokens"]) for row in steady
        )
        / sum(steady_times),
        "peak_gpu_memory_mb_max": max(
            float(row["peak_gpu_memory_mb"]) for row in rows
        ),
        "allocated_gpu_memory_mb_max": max(
            float(row["allocated_gpu_memory_mb"]) for row in rows
        ),
        "reserved_gpu_memory_mb_max": max(
            float(row["reserved_gpu_memory_mb"]) for row in rows
        ),
    }


def load_runs(raw: Path, expected_steps: int):
    runs = defaultdict(dict)
    for source in sorted(raw.glob("training_figure4_gemma_*_seed*.csv")):
        match = RUN_NAME.fullmatch(source.name)
        if match is None:
            continue
        implementation, seed_text = match.groups()
        rows = read_training_rows(source)
        if len(rows) != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} steps in {source}, found {len(rows)}"
            )
        runs[implementation][int(seed_text)] = rows
    return runs


def write_run_summaries(runs, implementations, output: Path) -> None:
    summaries = []
    for implementation in implementations:
        for seed, rows in sorted(runs[implementation].items()):
            summary = run_summary(implementation, rows)
            summaries.append(
                {"implementation": implementation, "seed": seed}
                | {key: value for key, value in summary.items() if key != "implementation"}
            )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def build_curves(runs, implementations, steps: int, window: int):
    t_critical_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    curve_rows, mean_curves = [], {}
    for implementation in implementations:
        seed_rows = list(runs[implementation].values())
        mean_curve = []
        for index in range(steps):
            losses = [float(rows[index]["train_loss"]) for rows in seed_rows]
            smoothed = [
                statistics.fmean(
                    float(row["train_loss"])
                    for row in rows[max(0, index - window + 1) : index + 1]
                )
                for rows in seed_rows
            ]
            critical = t_critical_95[len(losses)]
            mean = statistics.fmean(losses)
            half_width = critical * statistics.stdev(losses) / math.sqrt(len(losses))
            smooth_mean = statistics.fmean(smoothed)
            smooth_half_width = (
                critical * statistics.stdev(smoothed) / math.sqrt(len(smoothed))
            )
            mean_curve.append(mean)
            curve_rows.append(
                {
                    "implementation": implementation,
                    "step": index + 1,
                    "n_seeds": len(losses),
                    "mean_loss": mean,
                    "std_loss": statistics.stdev(losses),
                    "ci95_low": mean - half_width,
                    "ci95_high": mean + half_width,
                    "smoothing_window": window,
                    "mean_loss_smoothed": smooth_mean,
                    "ci95_low_smoothed": smooth_mean - smooth_half_width,
                    "ci95_high_smoothed": smooth_mean + smooth_half_width,
                }
            )
        mean_curves[implementation] = mean_curve
    return curve_rows, mean_curves


def pair_summary(runs, curves, seeds: list[int], steps: int, confidence: float):
    for seed in seeds:
        cce_tokens = [float(row["tokens"]) for row in runs["cce"][seed]]
        control_tokens = [float(row["tokens"]) for row in runs["torch_compile"][seed]]
        if cce_tokens != control_tokens:
            raise ValueError(f"Figure 4 token counts differ within seed pair {seed}")

    cce_curve, control_curve = curves["cce"], curves["torch_compile"]
    deltas = [
        cce - control
        for cce, control in zip(cce_curve, control_curve, strict=True)
    ]
    seed_deltas = {
        seed: statistics.fmean(
            float(cce["train_loss"]) - float(control["train_loss"])
            for cce, control in zip(
                runs["cce"][seed][-50:],
                runs["torch_compile"][seed][-50:],
                strict=True,
            )
        )
        for seed in seeds
    }
    values = list(seed_deltas.values())
    delta_mean = statistics.fmean(values)
    half_width = 2.776 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "model": "google/gemma-2-2b-it",
        "implementations": ["cce", "torch_compile"],
        "seeds": seeds,
        "steps": steps,
        "confidence_level": confidence,
        "paired_token_counts_match": True,
        "mean_curve_pearson_r": statistics.correlation(cce_curve, control_curve),
        "mean_curve_rmse": math.sqrt(statistics.fmean(delta**2 for delta in deltas)),
        "mean_curve_mae": statistics.fmean(abs(delta) for delta in deltas),
        "mean_curve_max_abs_diff": max(abs(delta) for delta in deltas),
        "cce_last50_mean_loss": statistics.fmean(cce_curve[-50:]),
        "torch_compile_last50_mean_loss": statistics.fmean(control_curve[-50:]),
        "last50_mean_delta_cce_minus_torch_compile": statistics.fmean(deltas[-50:]),
        "paired_seed_last50_mean_deltas": seed_deltas,
        "paired_seed_last50_delta_mean": delta_mean,
        "paired_seed_last50_delta_ci95_low": delta_mean - half_width,
        "paired_seed_last50_delta_ci95_high": delta_mean + half_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    parser.add_argument(
        "--processed-dir", default=str(PROJECT_ROOT / "results" / "processed")
    )
    args = parser.parse_args()
    raw, processed = Path(args.raw_dir), Path(args.processed_dir)
    config = json.loads((PROJECT_ROOT / "configs" / "experiments.json").read_text())
    experiment = config["figure4_gemma_training"]
    implementations = experiment["implementations"]
    seeds = sorted(int(seed) for seed in experiment["seeds"])
    steps = int(experiment["max_steps"])
    runs = load_runs(raw, steps)
    for implementation in implementations:
        if set(runs[implementation]) != set(seeds):
            raise ValueError(f"Incomplete Figure 4 runs for {implementation}")

    processed.mkdir(parents=True, exist_ok=True)
    runs_output = processed / "figure4_gemma_runs_summary.csv"
    write_run_summaries(runs, implementations, runs_output)
    print(f"Wrote {runs_output}")

    curve_rows, curves = build_curves(
        runs,
        implementations,
        steps,
        int(experiment["plot_smoothing_window"]),
    )
    curves_output = processed / "figure4_gemma_curves.csv"
    with curves_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(f"Wrote {curves_output}")

    pair = pair_summary(runs, curves, seeds, steps, experiment["confidence_level"])
    pair_output = processed / "figure4_gemma_pair_summary.json"
    pair_output.write_text(json.dumps(pair, indent=2) + "\n")
    print(f"Wrote {pair_output}")


if __name__ == "__main__":
    main()
