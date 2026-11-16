"""Compare buffer sizes using prediction-before-learning evaluation."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from benchmark.run_stream_vectorization import git_output
from numcompute_stream._memory import array_storage_bytes
from numcompute_stream.io import iter_csv_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "demo/data/stream_classification.csv"
DATA_SHA256 = (
    "38451bd12d42d19fd1aa2214a3688d5fc379"
    "acec86853519719c7fe642f27e08"
)
BUFFERS = (100, 300, 600)
CHUNK_SIZE = 50
COMMON_START = 600  # Zero-based: original data rows 601 onward.


def run_once(buffer_size):
    model = Pipeline([
        ("imputer", Imputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(
            buffer_size=buffer_size,
            max_depth=5,
            min_samples_split=2,
            max_features=None,
            criterion="gini",
            random_state=2026,
        )),
    ])
    seen = evaluated = correct = common_n = common_correct = 0
    predict_seconds = 0.0
    fit_seconds = 0.0
    events = []
    predictions = []
    sampled_bytes = [array_storage_bytes(model)]
    max_pending_rows = 0

    # This duration includes CSV reading and diagnostic bookkeeping.
    start_flow = perf_counter()
    for chunk_id, (X, y) in enumerate(iter_csv_chunks(
        DATA,
        target_column="target",
        feature_columns=["x1", "x2", "x3"],
        label_dtype="int",
        chunk_size=CHUNK_SIZE,
    ), start=1):
        assert len(y) == CHUNK_SIZE
        before_version = model.model_version_
        before_retrains = model.n_retrains_
        before_buffer = model.n_samples_buffered_
        ready = model.is_fitted_
        chunk_correct = None

        if ready:
            start = perf_counter()
            predicted = model.predict(X)
            predict_seconds += perf_counter() - start

            matches = predicted == y
            chunk_correct = int(np.count_nonzero(matches))
            evaluated += len(y)
            correct += chunk_correct

            common_mask = np.arange(seen, seen + len(y)) >= COMMON_START
            common_n += int(np.count_nonzero(common_mask))
            common_correct += int(np.count_nonzero(matches[common_mask]))
            predictions.extend(int(value) for value in predicted)

        if seen >= COMMON_START:
            assert ready, "A model is unavailable in the common interval."

        # Prediction must not train or change buffering.
        assert model.model_version_ == before_version
        assert model.n_retrains_ == before_retrains
        assert model.n_samples_buffered_ == before_buffer

        pending_rows = before_buffer + len(y)
        max_pending_rows = max(max_pending_rows, pending_rows)

        start = perf_counter()
        model.partial_fit(X, y)
        elapsed = perf_counter() - start
        fit_seconds += elapsed

        retrained = model.n_retrains_ == before_retrains + 1
        assert retrained == (pending_rows >= buffer_size)
        assert model.n_retrains_ == before_retrains + int(retrained)
        assert model.model_version_ == before_version + int(retrained)
        assert model.n_samples_buffered_ == (0 if retrained else pending_rows)
        trained_rows = int(model.n_samples_fit_) if retrained else 0
        if retrained:
            assert trained_rows == pending_rows

        seen += len(y)
        state_bytes = array_storage_bytes(model)
        sampled_bytes.append(state_bytes)
        events.append({
            "chunk": chunk_id,
            "last_data_row": seen,
            "evaluated": ready,
            "correct": chunk_correct,
            "model_version_before": before_version,
            "retrained": retrained,
            "trained_rows": trained_rows,
            "remaining_buffer_rows": model.n_samples_buffered_,
            "partial_fit_seconds": elapsed,
            "pipeline_array_bytes_after": state_bytes,
        })

    flow_seconds = perf_counter() - start_flow
    assert seen == 1200
    assert evaluated == 1200 - buffer_size
    assert common_n == 600
    assert model.n_retrains_ == 1200 // buffer_size
    assert model.n_samples_buffered_ == 0

    retrain_times = [
        event["partial_fit_seconds"] for event in events
        if event["retrained"]
    ]
    summary = {
        "buffer_size": buffer_size,
        "warmup_samples": seen - evaluated,
        "evaluated_samples": evaluated,
        "correct": correct,
        "accuracy": correct / evaluated,
        "common_samples": common_n,
        "common_correct": common_correct,
        "common_accuracy": common_correct / common_n,
        "retrain_count": model.n_retrains_,
        "retrain_call_total_seconds": sum(retrain_times),
        "retrain_call_median_seconds": median(retrain_times),
        "all_partial_fit_seconds": fit_seconds,
        "prediction_call_seconds": predict_seconds,
        "instrumented_flow_seconds": flow_seconds,
        "max_pending_rows": max_pending_rows,
        "max_sampled_pipeline_array_bytes": max(sampled_bytes),
        "final_pipeline_array_bytes": sampled_bytes[-1],
    }
    return {
        "summary": summary,
        "events": events,
        "prelearning_predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Run without -O so checks execute.")

    actual_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if actual_hash != DATA_SHA256:
        raise ValueError("CSV hash differs from the verified dataset.")

    repeats = 1 if args.quick else 7
    runs = []
    baselines = {}

    # Formal runs warm up each configuration using a disposable model.
    if not args.quick:
        for buffer_size in BUFFERS:
            run_once(buffer_size)

    for repeat in range(repeats):
        offset = repeat % len(BUFFERS)
        order = BUFFERS[offset:] + BUFFERS[:offset]
        for buffer_size in order:
            result = run_once(buffer_size)
            summary = result["summary"]
            summary["repeat"] = repeat + 1

            # Timing varies; historical predictions and event structure
            # must reproduce exactly across fresh models.
            signature = (
                result["prelearning_predictions"],
                [
                    (
                        event["chunk"], event["evaluated"],
                        event["correct"], event["retrained"],
                        event["trained_rows"],
                        event["remaining_buffer_rows"],
                    )
                    for event in result["events"]
                ],
            )
            if buffer_size in baselines:
                assert signature == baselines[buffer_size]
            else:
                baselines[buffer_size] = signature

            runs.append(result)
            print(
                f"repeat={repeat + 1} buffer={buffer_size:3d} "
                f"eval={summary['evaluated_samples']:4d} "
                f"accuracy={summary['accuracy']:.6f} "
                f"common={summary['common_accuracy']:.6f} "
                f"retrains={summary['retrain_count']:2d} "
                f"retrain_total="
                f"{summary['retrain_call_total_seconds']:.6f}s"
            )

    sources = sorted((ROOT / "numcompute_stream").glob("*.py"))
    sources += [
        Path(__file__).resolve(),
        ROOT / "benchmark/run_stream_vectorization.py",
        ROOT / "benchmark/stream_references.py",
    ]
    metadata = {
        "mode": "smoke" if args.quick else "formal",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "command_arguments": sys.argv,
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "data_sha256": actual_hash,
        "source_sha256": {
            path.relative_to(ROOT).as_posix():
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "buffer_sizes": BUFFERS,
        "chunk_size": CHUNK_SIZE,
        "common_data_rows_inclusive": [601, 1200],
        "repeats": repeats,
        "warmup_runs_per_configuration": 0 if args.quick else 1,
        "tree": {
            "max_depth": 5, "min_samples_split": 2,
            "max_features": None, "criterion": "gini",
            "random_state": 2026,
        },
        "preprocessing": ["Imputer(mean)", "StandardScaler"],
        "timing_scope": (
            "Retrain time is the entire triggering Pipeline.partial_fit "
            "call, including validation, buffering, preprocessing, tree "
            "fit and publication. It is not isolated tree fit time. "
            "Instrumented flow includes CSV reading, predictions, "
            "training, assertions, counting and memory sampling; "
            "excludes model creation and result-file writing."
        ),
        "memory_scope": (
            "Logical reachable ndarray bytes of Pipeline, sampled before "
            "the stream and after updates. Excludes temporary allocations, "
            "Python overhead and benchmark logs/predictions. Not peak RAM."
        ),
        "evaluation": (
            "Predict before learning; warm-up excluded. Accuracy is "
            "correct-count / sample-count. Common interval continues "
            "online learning. No tail flush."
        ),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "benchmark/results" / (
        f"stream_buffers_{metadata['mode']}_{stamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(runs[0]["summary"])
        )
        writer.writeheader()
        writer.writerows(run["summary"] for run in runs)

    with (output / "raw_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "runs": runs},
            handle, indent=2, allow_nan=False,
        )
        handle.write("\n")

    print("\nBuffer benchmark checks: PASS")
    print(f"Mode: {metadata['mode']}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()