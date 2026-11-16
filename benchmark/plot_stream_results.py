"""Plot recorded streaming results without rerunning experiments."""

import hashlib
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark/results"

SOURCES = {
    "vectorization": RESULTS / (
        "stream_vectorization_formal_20261116T104755_980073Z"
    ) / "raw_results.json",
    "buffers": RESULTS / (
        "stream_buffers_formal_20261116T110038_916524Z"
    ) / "raw_results.json",
    "growth": RESULTS / (
        "stream_state_growth_20261116T110353_331858Z"
    ) / "raw_results.json",
}
OUTPUT = RESULTS / "stream_summary_figures"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig, filename):
    path = OUTPUT / filename
    try:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)

    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    print(f"Saved: {path}")
    return path


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so checks execute.")

    vector = load_json(SOURCES["vectorization"])
    buffers = load_json(SOURCES["buffers"])
    growth = load_json(SOURCES["growth"])

    assert vector["metadata"]["mode"] == "formal"
    assert buffers["metadata"]["mode"] == "formal"
    assert len(vector["cases"]) == 9
    assert len(buffers["runs"]) == 21

    OUTPUT.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Implementation speedups, including results below 1x.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    confusion = sorted(
        (
            case for case in vector["cases"]
            if case["operation"] == "confusion"
        ),
        key=lambda case: case["n_samples"],
    )
    assert len(confusion) == 3
    axes[0].plot(
        [case["n_samples"] for case in confusion],
        [case["speedup"] for case in confusion],
        marker="o",
        label="Full confusion-matrix operation",
    )

    for criterion in ("gini", "entropy"):
        cases = sorted(
            (
                case for case in vector["cases"]
                if case["operation"] == "split_search"
                and case["criterion"] == criterion
            ),
            key=lambda case: case["n_samples"],
        )
        assert len(cases) == 3
        axes[1].plot(
            [case["n_samples"] for case in cases],
            [case["speedup"] for case in cases],
            marker="o",
            label=criterion.capitalize(),
        )

    for ax, title in zip(axes, ("Confusion matrix", "Decision-tree split search")):
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Samples (log scale)")
        ax.set_ylabel("Loop / vectorised median time")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Measured implementation speedups; below 1x is slower")
    fig.tight_layout()
    generated.append(save_figure(fig, "vectorization_speedups.png"))

    # 2. Common-interval accuracy and per-run total retraining times.
    summaries = [run["summary"] for run in buffers["runs"]]
    sizes = sorted({row["buffer_size"] for row in summaries})
    assert sizes == [100, 300, 600]

    grouped = {
        size: [row for row in summaries if row["buffer_size"] == size]
        for size in sizes
    }
    accuracy = []
    times = []
    table = []

    for size in sizes:
        rows = grouped[size]
        assert len(rows) == 7
        assert {row["repeat"] for row in rows} == set(range(1, 8))
        assert all(row["common_samples"] == 600 for row in rows)
        assert len({row["common_correct"] for row in rows}) == 1

        common_accuracy = rows[0]["common_correct"] / 600
        run_times = [
            row["retrain_call_total_seconds"] for row in rows
        ]
        accuracy.append(common_accuracy * 100)
        times.append(run_times)
        table.append({
            "buffer_size": size,
            "common_samples": 600,
            "common_accuracy": common_accuracy,
            "retrain_count": rows[0]["retrain_count"],
            "median_total_retrain_call_seconds": median(run_times),
            "min_total_retrain_call_seconds": min(run_times),
            "max_total_retrain_call_seconds": max(run_times),
        })

    positions = list(range(len(sizes)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    bars = axes[0].bar(positions, accuracy, color="#377eb8")
    axes[0].bar_label(
        bars, labels=[f"{value:.2f}%" for value in accuracy], padding=3
    )
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Common rows 601-1200; predict before learning")

    offsets = [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12]
    for position, values in zip(positions, times):
        axes[1].scatter(
            [position + offset for offset in offsets],
            [value * 1000 for value in values],
            color="#777777", alpha=0.75,
            label="Individual run" if position == 0 else None,
        )
    axes[1].plot(
        positions,
        [median(values) * 1000 for values in times],
        "D-", color="#d95f02", label="Median of 7 runs",
    )
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("Total retraining-call time per run (ms)")
    axes[1].set_title("Entire triggering Pipeline.partial_fit calls")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(positions, [str(size) for size in sizes])
        ax.set_xlabel("Buffer size")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.tight_layout()
    generated.append(save_figure(fig, "buffer_comparison.png"))

    # 3. Array storage only; logarithmic y-axis exposes all four series.
    checkpoints = sorted(growth["checkpoints"], key=lambda row: row["samples_seen"])
    assert len(checkpoints) == 4

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for key, label in (
        ("confusion_array_bytes", "Confusion matrix"),
        ("cumulative_auc_array_bytes", "Cumulative AUC"),
        ("rolling_auc_array_bytes", "Rolling AUC, window=500"),
        ("stats_array_bytes", "Stats with exact-quantile history"),
    ):
        values = [row[key] for row in checkpoints]
        assert all(value > 0 for value in values)
        ax.plot(
            [row["samples_seen"] for row in checkpoints],
            values, marker="o", label=label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Samples received")
    ax.set_ylabel("Reachable ndarray bytes (log scale)")
    ax.set_title("Retained component arrays; not process or peak RAM")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    generated.append(save_figure(fig, "retained_array_growth.png"))

    # Preserve exact raw-data aggregates and provenance for these figures.
    with (OUTPUT / "buffer_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    manifest = {
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in SOURCES.items()
        },
        "plot_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "figures": [
            path.relative_to(ROOT).as_posix() for path in generated
        ],
        "note": (
            "Figures use recorded results; benchmarks were not rerun. "
            "Repeated plotting overwrites these derived outputs."
        ),
    }
    with (OUTPUT / "sources.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nExact buffer timing medians from raw results:")
    for row in table:
        print(
            f"buffer={row['buffer_size']:3d} "
            f"common={row['common_accuracy']:.6f} "
            f"median_total="
            f"{row['median_total_retrain_call_seconds']:.9f}s"
        )
    print("\nResult figure checks: PASS (3 PNG files)")


if __name__ == "__main__":
    main()