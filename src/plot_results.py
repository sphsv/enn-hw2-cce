from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLORS = {"baseline": "#d55e00", "torch_compile": "#0072b2", "cce": "#009e73"}
METHODS = ("baseline", "torch_compile", "cce")
TABLE1_ORDER = (
    "cce",
    "liger",
    "torchtune_8_chunks",
    "torch_compile",
    "baseline",
    "cce_no_grad_filter",
    "cce_kahan",
    "cce_kahan_full_c",
    "cce_kahan_full_e",
)
TABLE1_LABELS = {
    "cce": "CCE",
    "liger": "Liger Kernels",
    "torchtune_8_chunks": "TorchTune 0.3.1 (8 chunks)",
    "torch_compile": "torch.compile",
    "baseline": "Baseline",
    "cce_no_grad_filter": "CCE — no grad filter",
    "cce_kahan": "CCE-Kahan/FP32",
    "cce_kahan_full_c": "CCE-Kahan-FullC",
    "cce_kahan_full_e": "CCE-Kahan-FullE",
}


def scaling_plot(data: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    operations = list(data["operation"].unique())
    figure, axes = plt.subplots(1, len(operations), figsize=(5 * len(operations), 4), squeeze=False)
    for axis, operation in zip(axes[0], operations, strict=True):
        subset = data[data["operation"] == operation]
        for method in METHODS:
            method_data = subset[subset["method"] == method].sort_values("vocab_size")
            if method_data.empty:
                continue
            axis.plot(
                method_data["vocab_size"],
                method_data[metric],
                marker="o",
                label=method,
                color=COLORS[method],
            )
        axis.set_title(operation.replace("_", " + "))
        axis.set_xlabel("Vocabulary size")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0][0].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def official_bars(data: pd.DataFrame, output: Path, title: str) -> None:
    operations = ("loss-fw", "loss-bw", "loss-fw-bw")
    labels = ("forward", "backward", "forward + backward")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    width = 0.24
    x = list(range(len(operations)))
    for method_index, method in enumerate(METHODS):
        subset = data[data["method"] == method].set_index("kind")
        offset = (method_index - 1) * width
        axes[0].bar(
            [value + offset for value in x],
            [subset.loc[operation, "runtime_ms_median"] for operation in operations],
            width,
            label=method,
            color=COLORS[method],
        )
        axes[1].bar(
            [value + offset for value in x],
            [subset.loc[operation, "op_mem_mb_median"] for operation in operations],
            width,
            label=method,
            color=COLORS[method],
        )
    axes[0].set_ylabel("Runtime, ms")
    axes[1].set_ylabel("Peak operation memory, MB (log scale)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def filtering_plot(summary: pd.DataFrame, correctness_path: Path, output: Path) -> None:
    correctness = json.loads(correctness_path.read_text())["cases"]
    order = ("unfiltered", "filtered_auto")
    labels = ("unfiltered", "filtered (auto)")
    indexed = summary.set_index("variant")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(
        labels,
        [indexed.loc[name, "runtime_ms_median"] for name in order],
        color=("#999999", COLORS["cce"]),
    )
    axes[0].set_ylabel("Backward runtime, ms")
    axes[0].set_title("Filtering speed")
    seeds = [case["seed"] for case in correctness]
    axes[1].plot(
        seeds,
        [case["embedding_gradient"]["relative_l2_error"] for case in correctness],
        marker="o",
        label="embedding gradient",
    )
    axes[1].plot(
        seeds,
        [case["classifier_gradient"]["relative_l2_error"] for case in correctness],
        marker="o",
        label="classifier gradient",
    )
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("Relative L2 error vs unfiltered")
    axes[1].set_title("Filtering numerical error")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def table1_extended_plot(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["kind"] == "loss-fw-bw"].set_index("method").loc[list(TABLE1_ORDER)]
    labels = [TABLE1_LABELS[method] for method in TABLE1_ORDER]
    colors = [
        COLORS.get(method, "#7b6fd0")
        if not method.startswith("cce_")
        else COLORS["cce"]
        for method in TABLE1_ORDER
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    axes[0].barh(labels, data["runtime_ms_median"], color=colors)
    axes[0].set_xlabel("Loss + gradient runtime, ms")
    axes[0].invert_yaxis()
    axes[1].barh(labels, data["op_mem_mb_median"], color=colors)
    axes[1].set_xlabel("Operation memory, MB (log scale)")
    axes[1].set_xscale("log")
    axes[1].invert_yaxis()
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Extended Table 1 reproduction — Gemma-2 exact shape")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def figure3_plot(raw: pd.DataFrame, output: Path) -> None:
    threshold = float(raw["filter_threshold"].iloc[0])
    below = raw[raw["average_probability"] < threshold]
    crossing = int(below["rank"].iloc[0]) if not below.empty else None
    figure, axis = plt.subplots(figsize=(7, 4.6))
    axis.loglog(raw["rank"], raw["average_probability"], color=COLORS["cce"])
    axis.axhline(threshold, color="#d55e00", linestyle="--", label="BF16 ε / 32")
    if crossing is not None:
        axis.axvline(
            crossing,
            color="#666666",
            linestyle=":",
            label=f"first below threshold: rank {crossing}",
        )
    axis.set_xlabel("Token probability rank")
    axis.set_ylabel("Average softmax probability")
    axis.set_title("Figure 3 reproduction — probability decay")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def figure4_gemma_plot(curves: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for implementation in ("torch_compile", "cce"):
        subset = curves[curves["implementation"] == implementation].sort_values("step")
        x = subset["step"].to_numpy()
        mean = subset["mean_loss_smoothed"].to_numpy()
        low = subset["ci95_low_smoothed"].to_numpy()
        high = subset["ci95_high_smoothed"].to_numpy()
        axis.plot(x, mean, label=implementation, color=COLORS[implementation])
        axis.fill_between(x, low, high, color=COLORS[implementation], alpha=0.18)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Mean training loss")
    window = int(curves["smoothing_window"].iloc[0])
    axis.set_title(
        f"Gemma-2 fine-tuning — {window}-step rolling mean, 5 seeds, 95% CI"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir", default=str(PROJECT_ROOT / "results" / "processed")
    )
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "results" / "raw"))
    parser.add_argument("--figure-dir", default=str(PROJECT_ROOT / "results" / "figures"))
    args = parser.parse_args()
    processed, raw, figures = Path(args.processed_dir), Path(args.raw_dir), Path(args.figure_dir)
    figures.mkdir(parents=True, exist_ok=True)

    scaling_path = processed / "scaling_summary.csv"
    if not scaling_path.exists():
        raise SystemExit(f"Missing {scaling_path}; run python -m src.summarize first")
    data = pd.read_csv(scaling_path)
    scaling_plot(
        data,
        "peak_memory_mb_median",
        "Peak operation memory, MB",
        figures / "memory_vs_vocabulary.png",
    )
    scaling_plot(
        data,
        "runtime_ms_median",
        "Runtime, ms",
        figures / "runtime_vs_vocabulary.png",
    )
    exact_path = processed / "official_benchmark_summary.csv"
    if exact_path.exists():
        official_bars(
            pd.read_csv(exact_path),
            figures / "gemma_exact_comparison.png",
            "Exact Gemma-2 + Alpaca benchmark (official harness, 50 × 3)",
        )
    filtering_path = processed / "filtering_summary.csv"
    filtering_correctness = raw / "filtering_correctness.json"
    if filtering_path.exists() and filtering_correctness.exists():
        filtering_plot(
            pd.read_csv(filtering_path),
            filtering_correctness,
            figures / "gradient_filtering_ablation.png",
        )
    table1_path = processed / "table1_extended_summary.csv"
    if table1_path.exists():
        table1_extended_plot(
            pd.read_csv(table1_path), figures / "table1_extended_comparison.png"
        )
    figure3_path = raw / "figure3_probabilities.csv"
    if figure3_path.exists():
        figure3_plot(
            pd.read_csv(figure3_path), figures / "figure3_probability_decay.png"
        )
    figure4_path = processed / "figure4_gemma_curves.csv"
    if figure4_path.exists():
        figure4_gemma_plot(
            pd.read_csv(figure4_path), figures / "figure4_gemma_finetuning.png"
        )
    print(f"Wrote figures to {figures}")


if __name__ == "__main__":
    main()
