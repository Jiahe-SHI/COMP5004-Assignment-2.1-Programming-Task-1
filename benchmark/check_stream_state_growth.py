"""Inspect retained array storage across a longer synthetic stream."""

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark.run_stream_vectorization import git_output
from numcompute_stream._memory import array_storage_bytes
from numcompute_stream.metrics import (
    RollingROCAUC,
    StreamingConfusionMatrix,
    StreamingROCAUC,
)
from numcompute_stream.stats import StreamingStats


ROOT = Path(__file__).resolve().parents[1]
TOTAL = 50000
CHUNK = 100
WINDOW = 500
SEED = 2026
CHECKPOINTS = (1000, 10000, 25000, 50000)


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so checks execute.")

    rng = np.random.default_rng(SEED)
    confusion = StreamingConfusionMatrix(classes=[0, 1])
    cumulative = StreamingROCAUC(positive_label=1)
    rolling = RollingROCAUC(WINDOW, positive_label=1)
    stats = StreamingStats()
    rows = []

    for start in range(0, TOTAL, CHUNK):
        X = rng.normal(size=(CHUNK, 3))
        # Alternating labels guarantee both classes in every chunk.
        y = (np.arange(start, start + CHUNK) % 2).astype(np.int64)
        scores = X[:, 0] + 0.75 * y
        predicted = (scores > 0.375).astype(np.int64)

        confusion.update(y, predicted)
        cumulative.update(y, scores)
        rolling.update(y, scores)
        stats.update_stats(X)

        seen = start + CHUNK
        assert confusion.n_samples_seen_ == seen
        assert cumulative.n_samples_seen_ == seen
        assert rolling.n_samples_seen_ == seen
        assert stats.n_rows_seen_ == seen

        if seen not in CHECKPOINTS:
            continue

        cm_result = confusion.result()
        cumulative_result = cumulative.result()
        rolling_result = rolling.result()
        stats_result = stats.result()

        assert int(cm_result["confusion_matrix"].sum()) == seen
        assert cumulative_result["n_samples"] == seen
        assert rolling_result["n_samples"] == min(WINDOW, seen)
        np.testing.assert_array_equal(stats_result["count"], [seen, seen, seen])
        assert np.isfinite(cumulative_result["auc"])
        assert np.isfinite(rolling_result["auc"])

        row = {
            "samples_seen": seen,
            "confusion_array_bytes": array_storage_bytes(confusion),
            "cumulative_auc_array_bytes": array_storage_bytes(cumulative),
            "rolling_auc_array_bytes": array_storage_bytes(rolling),
            "stats_array_bytes": array_storage_bytes(stats),
            "cumulative_auc_retained_samples":
                cumulative_result["n_samples"],
            "rolling_auc_retained_samples":
                rolling_result["n_samples"],
            "cumulative_auc": cumulative_result["auc"],
            "rolling_auc": rolling_result["auc"],
        }
        rows.append(row)
        print(
            f"n={seen:6d} "
            f"confusion={row['confusion_array_bytes']:8d} "
            f"cumulative_auc={row['cumulative_auc_array_bytes']:8d} "
            f"rolling_auc={row['rolling_auc_array_bytes']:8d} "
            f"stats={row['stats_array_bytes']:8d} bytes"
        )

    assert len(rows) == len(CHECKPOINTS)

    # Fixed class count; window already full at every checkpoint.
    assert len({row["confusion_array_bytes"] for row in rows}) == 1
    assert len({row["rolling_auc_array_bytes"] for row in rows}) == 1

    for previous, current in zip(rows, rows[1:]):
        assert (
            current["cumulative_auc_array_bytes"]
            > previous["cumulative_auc_array_bytes"]
        )
        assert current["stats_array_bytes"] > previous["stats_array_bytes"]

    # Exercise the exact-quantile query after the longer stream.
    quantiles = stats.quantile(0.5)
    assert quantiles.shape == (3,)
    assert np.isfinite(quantiles).all()

    source_paths = sorted((ROOT / "numcompute_stream").glob("*.py"))
    source_paths += [
        Path(__file__).resolve(),
        ROOT / "benchmark/run_stream_vectorization.py",
        ROOT / "benchmark/stream_references.py",
    ]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": {
            path.relative_to(ROOT).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "seed": SEED,
        "total_samples": TOTAL,
        "chunk_size": CHUNK,
        "window_size": WINDOW,
        "checkpoints": CHECKPOINTS,
        "data": (
            "Generated per chunk: three standard-normal features, "
            "alternating binary labels, score=X[:,0]+0.75*y, "
            "predicted label=int(score>0.375). No missing features."
        ),
        "memory_method": "reachable_ndarray_nbytes",
        "scope": (
            "Retained arrays reachable from each component separately. "
            "Excludes Python object overhead, object-array referents, "
            "temporary allocations, input chunks and output records. "
            "Not process RAM, peak memory or a speed benchmark."
        ),
        "final_feature_medians": quantiles.tolist(),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "benchmark/results" / f"stream_state_growth_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    with (output / "checkpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output / "raw_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "checkpoints": rows},
            handle, indent=2, allow_nan=False,
        )
        handle.write("\n")

    print("\nState growth checks: PASS")
    print("Measurement: retained ndarray bytes, not process or peak RAM")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()