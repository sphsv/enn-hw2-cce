from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize_csv(
    source: Path,
    destination: Path,
    group_fields: list[str],
    value_fields: list[str],
) -> None:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status", "ok") == "ok":
                groups[tuple(row[field] for field in group_fields)].append(row)

    output_fields = group_fields + [
        f"{value}_{stat}"
        for value in value_fields
        for stat in ("count", "mean", "median", "std", "min", "max")
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            output = dict(zip(group_fields, key, strict=True))
            for value_field in value_fields:
                values = [float(row[value_field]) for row in rows if row.get(value_field)]
                output |= {
                    f"{value_field}_count": len(values),
                    f"{value_field}_mean": statistics.fmean(values),
                    f"{value_field}_median": statistics.median(values),
                    f"{value_field}_std": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                    f"{value_field}_min": min(values),
                    f"{value_field}_max": max(values),
                }
            writer.writerow(output)


def table1_paper_comparison(processed: Path) -> None:
    with (processed / "table1_extended_summary.csv").open(newline="") as handle:
        reproduction = {
            (row["method"], row["kind"]): row for row in csv.DictReader(handle)
        }

    output_rows = []
    paper_values = PROJECT_ROOT / "configs" / "paper_table1.csv"
    with paper_values.open(newline="") as handle:
        for paper in csv.DictReader(handle):
            row = reproduction[(paper["method"], paper["kind"])]
            paper_runtime = (
                float(paper["paper_runtime_ms"])
                if paper["paper_runtime_ms"]
                else None
            )
            paper_memory = (
                float(paper["paper_op_mem_mb"])
                if paper["paper_op_mem_mb"]
                else None
            )
            ours_runtime = float(row["runtime_ms_median"])
            ours_memory = float(row["op_mem_mb_median"])
            output_rows.append(
                {
                    "method": paper["method"],
                    "kind": paper["kind"],
                    "paper_runtime_ms": paper_runtime,
                    "reproduction_runtime_ms": ours_runtime,
                    "runtime_relative_difference_pct": (
                        None
                        if paper_runtime is None
                        else 100.0 * (ours_runtime / paper_runtime - 1.0)
                    ),
                    "paper_op_mem_mb": paper_memory,
                    "reproduction_op_mem_mb": ours_memory,
                    "memory_relative_difference_pct": (
                        None
                        if paper_memory is None
                        else 100.0 * (ours_memory / paper_memory - 1.0)
                    ),
                    "timing_semantics": row["timing_semantics"],
                }
            )

    output = processed / "table1_paper_comparison.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    parser.add_argument(
        "--processed-dir", default=str(PROJECT_ROOT / "results" / "processed")
    )
    args = parser.parse_args()
    raw, processed = Path(args.raw_dir), Path(args.processed_dir)
    jobs = [
        (
            "official_benchmark_reps.csv",
            "official_benchmark_summary.csv",
            ["method", "kind", "test_data"],
            ["runtime_ms", "op_mem_mb"],
        ),
        (
            "scaling.csv",
            "scaling_summary.csv",
            ["method", "operation", "vocab_size", "num_tokens", "hidden_dim", "dtype"],
            ["runtime_ms", "peak_memory_mb"],
        ),
        (
            "filtering.csv",
            "filtering_summary.csv",
            ["variant", "num_tokens", "vocab_size", "hidden_dim", "dtype"],
            ["runtime_ms", "peak_memory_mb"],
        ),
        (
            "table1_extended_reps.csv",
            "table1_extended_summary.csv",
            ["method", "kind", "test_data", "timing_semantics"],
            ["runtime_ms", "op_mem_mb"],
        ),
    ]
    for source_name, output_name, groups, values in jobs:
        source = raw / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        destination = processed / output_name
        summarize_csv(source, destination, groups, values)
        print(f"Wrote {destination}")
    table1_paper_comparison(processed)


if __name__ == "__main__":
    main()
