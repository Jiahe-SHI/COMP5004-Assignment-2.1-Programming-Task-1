"""Reproducible loop-versus-vectorised implementation benchmarks."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from benchmark.stream_references import (
    loop_best_split,
    loop_confusion_matrix,
)
from numcompute_stream.metrics import StreamingConfusionMatrix
from numcompute_stream.tree import _best_split


ROOT = Path(__file__).resolve().parents[1]


def git_output(*args):
    """Read Git metadata without requiring Git to be installed."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def vector_confusion(truth, prediction):
    metric = StreamingConfusionMatrix()
    metric.update(truth, prediction)
    return metric.result()


def check_confusion(expected, actual):
    np.testing.assert_array_equal(actual["classes"], expected["classes"])
    np.testing.assert_array_equal(
        actual["confusion_matrix"], expected["confusion_matrix"]
    )
    assert actual["n_samples"] == expected["n_samples"]


def check_split(X, expected, actual):
    if expected is None:
        assert actual is None
        return

    assert actual is not None
    assert actual["feature"] == expected["feature"]
    assert actual["n_left"] == expected["n_left"]
    assert actual["n_right"] == expected["n_right"]
    np.testing.assert_allclose(
        actual["gain"], expected["gain"], rtol=0, atol=1e-12
    )
    feature = expected["feature"]
    np.testing.assert_array_equal(
        X[:, feature] <= actual["threshold"],
        X[:, feature] <= expected["boundary"],
    )


def measure_pair(loop_fn, vector_fn, repeats):
    """Warm both paths, then alternate execution order."""
    loop_fn()
    vector_fn()
    samples = {"loop": [], "vectorized": []}
    functions = {"loop": loop_fn, "vectorized": vector_fn}

    for repeat in range(repeats):
        order = (
            ("loop", "vectorized")
            if repeat % 2 == 0
            else ("vectorized", "loop")
        )
        for name in order:
            start = perf_counter()
            result = functions[name]()
            elapsed = perf_counter() - start
            samples[name].append(elapsed)
            # Release the returned object outside the timed interval.
            del result

    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke run only; not formal performance evidence.",
    )
    args = parser.parse_args()

    if not __debug__:
        raise RuntimeError("Run without -O so correctness checks execute.")

    repeats = 3 if args.quick else 7
    confusion_sizes = (100, 1000) if args.quick else (1000, 10000, 100000)
    split_sizes = (32, 128) if args.quick else (100, 1000, 10000)
    seed = 2026
    n_classes = 4
    n_features = 4
    rows = []
    raw = []

    def record(name, n_samples, features, criterion, loop_fn,
               vector_fn, check):
        # Check the exact input used for this timing case.
        check(loop_fn(), vector_fn())
        samples = measure_pair(loop_fn, vector_fn, repeats)
        loop_seconds = median(samples["loop"])
        vector_seconds = median(samples["vectorized"])
        if vector_seconds <= 0:
            raise RuntimeError("Timer resolution is insufficient.")

        row = {
            "operation": name,
            "n_samples": n_samples,
            "n_features": features,
            "n_classes": n_classes,
            "criterion": criterion,
            "repeats": repeats,
            "loop_median_seconds": loop_seconds,
            "vectorized_median_seconds": vector_seconds,
            "speedup": loop_seconds / vector_seconds,
        }
        rows.append(row)
        raw.append({**row, "timings_seconds": samples})
        print(
            f"{name:16s} n={n_samples:7d} "
            f"{criterion:7s} "
            f"loop={loop_seconds:.6f}s "
            f"vector={vector_seconds:.6f}s "
            f"speedup={row['speedup']:.3f}x"
        )

    for n_samples in confusion_sizes:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, n_samples]))
        truth = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
        prediction = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
        truth.setflags(write=False)
        prediction.setflags(write=False)
        record(
            "confusion", n_samples, 0, "",
            lambda: loop_confusion_matrix(truth, prediction),
            lambda: vector_confusion(truth, prediction),
            check_confusion,
        )

    for n_samples in split_sizes:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, n_samples]))
        X = rng.normal(size=(n_samples, n_features))
        y = rng.integers(0, n_classes, size=n_samples, dtype=np.int64)
        X.setflags(write=False)
        y.setflags(write=False)

        for criterion in ("gini", "entropy"):
            record(
                "split_search", n_samples, n_features, criterion,
                lambda: loop_best_split(X, y, n_classes, criterion=criterion),
                lambda: _best_split(X, y, n_classes, criterion=criterion),
                lambda expected, actual: check_split(X, expected, actual),
            )

    source_files = (
        "benchmark/run_stream_vectorization.py",
        "benchmark/stream_references.py",
        "numcompute_stream/metrics.py",
        "numcompute_stream/tree.py",
        "numcompute_stream/_validation.py",
    )
    metadata = {
        "mode": "smoke" if args.quick else "formal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_files
        },
        "seed": seed,
        "warmup_calls_per_implementation": 1,
        "repeats": repeats,
        "timing_scope": {
            "confusion": (
                "Batch loop reference versus fresh metric construction, "
                "one update and result copy; not an isolated kernel test."
            ),
            "split_search": (
                "Validated all-feature split search, including sorting; "
                "loop boundary and production midpoint are compared "
                "by their observed-sample partitions."
            ),
            "excluded": "Data generation, equivalence checks and file I/O.",
        },
    }

    # Every invocation gets a new directory; previous results are preserved.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "benchmark" / "results" / (
        f"stream_vectorization_{metadata['mode']}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    with (output / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output / "raw_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {"metadata": metadata, "cases": raw},
            handle, indent=2, allow_nan=False,
        )
        handle.write("\n")

    print(f"\nEquivalence checks: PASS ({len(rows)} cases)")
    print(f"Mode: {metadata['mode']}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()