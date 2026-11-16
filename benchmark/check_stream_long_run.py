"""Check a longer stream, tail retention and invalid-input recovery."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark.run_stream_vectorization import git_output
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
TOTAL = 20037
CHUNK = 100
BUFFER = 300
SEED = 2026


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def model_state(model):
    return (
        model.is_fitted_,
        model.model_version_,
        model.n_retrains_,
        model.n_samples_buffered_,
    )


def evaluation_state(trainer):
    return (
        trainer.n_chunks_scored_,
        trainer.n_samples_scored_,
        trainer.n_samples_evaluated_,
        json.dumps(
            trainer.metrics_result(),
            default=json_default,
            sort_keys=True,
            allow_nan=False,
        ),
    )


def check_invalid_input(trainer, X, y):
    """Reject invalid input without changing model or evaluation state."""
    model = trainer.model
    bad_X = X.copy()
    bad_X[0, 0] = np.inf

    before_model = model_state(model)
    before_evaluation = evaluation_state(trainer)
    before_prediction = model.predict(X)
    before_logs = trainer.memory_result()["log_events"]

    for operation, expected_logs in (
        (trainer.score_chunk, before_logs),
        (trainer.fit_chunk, before_logs + 1),
    ):
        try:
            operation(bad_X, y)
        except ValueError:
            pass
        else:
            raise AssertionError("Infinite input was unexpectedly accepted.")

        assert model_state(model) == before_model
        assert evaluation_state(trainer) == before_evaluation
        np.testing.assert_array_equal(
            model.predict(X), before_prediction
        )
        assert trainer.memory_result()["log_events"] == expected_logs

    # Failed scores are omitted; failed learning attempts are logged.
    failure = trainer.logs_[-1]
    assert failure["event"] == "fit"
    assert failure["success"] is False
    assert failure["error_type"] == "ValueError"


def main():
    if not __debug__:
        raise RuntimeError("Run without -O so checks execute.")

    rng = np.random.default_rng(SEED)
    model = Pipeline([
        ("imputer", Imputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(
            buffer_size=BUFFER,
            max_depth=5,
            min_samples_split=2,
            max_features=None,
            criterion="gini",
            random_state=SEED,
        )),
    ])
    trainer = StreamTrainer(
        model, window_size=500, classes=[0, 1], positive_label=1
    )
    checkpoints = []
    recovery_checked = False
    chunks_seen = 0

    for start in range(0, TOTAL, CHUNK):
        size = min(CHUNK, TOTAL - start)
        X = rng.normal(size=(size, 3))
        y = ((X[:, 0] > 0.25) | (X[:, 1] < -0.5)).astype(np.int64)
        flips = rng.random(size) < 0.05
        y[flips] = 1 - y[flips]
        X[rng.random(X.shape) < 0.05] = np.nan

        if start == 10000:
            check_invalid_input(trainer, X, y)
            recovery_checked = True
            print("Invalid score/fit rejection checks: PASS")

        before = model_state(model)
        record = trainer.score_chunk(X, y)
        assert model_state(model) == before
        assert record["warm_up"] == (start < BUFFER)

        evaluated_before_fit = trainer.n_samples_evaluated_
        trainer.fit_chunk(X, y)
        assert trainer.n_samples_evaluated_ == evaluated_before_fit

        chunks_seen += 1
        seen = start + size
        assert trainer.n_chunks_scored_ == chunks_seen
        assert trainer.n_samples_scored_ == seen
        assert trainer.n_samples_evaluated_ == max(0, seen - BUFFER)
        assert model.n_retrains_ == seen // BUFFER
        assert model.model_version_ == seen // BUFFER
        assert model.n_samples_buffered_ == seen % BUFFER
        if model.is_fitted_:
            assert model.n_samples_fit_ == BUFFER

        memory = trainer.memory_result()
        expected_logs = 2 * chunks_seen + int(recovery_checked)
        assert memory["log_events"] == expected_logs
        assert memory["memory_method"] == "reachable_ndarray_nbytes"

        if seen in (1000, 10000, 20000, TOTAL):
            metrics = trainer.metrics_result()
            evaluated = max(0, seen - BUFFER)
            assert metrics["cumulative"]["n_samples"] == evaluated
            assert metrics["cumulative_auc"]["n_samples"] == evaluated
            assert metrics["rolling"]["n_samples"] == min(500, evaluated)
            assert metrics["rolling_auc"]["n_samples"] == min(500, evaluated)

            row = {
                "samples_seen": seen,
                "evaluated_samples": evaluated,
                "retrain_count": model.n_retrains_,
                "buffer_samples": model.n_samples_buffered_,
                "log_events": memory["log_events"],
                "reachable_array_bytes": memory["memory_bytes"],
                "buffer_array_bytes": memory["buffer_array_bytes"],
                "accuracy": metrics["cumulative"]["accuracy"],
                "auc": metrics["cumulative_auc"]["auc"],
            }
            checkpoints.append(row)
            print(
                f"n={seen:5d} evaluated={evaluated:5d} "
                f"retrains={row['retrain_count']:2d} "
                f"buffer={row['buffer_samples']:3d} "
                f"logs={row['log_events']:3d} "
                f"arrays={row['reachable_array_bytes']} bytes"
            )

    assert recovery_checked
    assert chunks_seen == 201
    assert model.n_retrains_ == 66
    assert model.n_samples_buffered_ == 237
    assert trainer.n_samples_evaluated_ == 19737

    logs = trainer.logs_
    assert len(logs) == 403
    assert [entry["event_id"] for entry in logs] == list(range(1, 404))
    failed = [entry for entry in logs if not entry["success"]]
    assert len(failed) == 1
    assert failed[0]["event"] == "fit"

    successful_events = [entry["event"] for entry in logs if entry["success"]]
    assert successful_events == ["score", "fit"] * chunks_seen
    assert all(entry["memory_status"] == "ok" for entry in logs)

    sources = sorted((ROOT / "numcompute_stream").glob("*.py"))
    sources += [
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
            for path in sources
        },
        "seed": SEED,
        "total_samples": TOTAL,
        "chunk_size": CHUNK,
        "buffer_size": BUFFER,
        "window_size": 500,
        "max_depth": 5,
        "data": (
            "Generated per chunk: standard-normal features; target is "
            "(x1>0.25) or (x2<-0.5), 5% label flips, then 5% missing "
            "features. Generated stream, not replay of the demo CSV."
        ),
        "scope": (
            "Functional long-stream check, not a performance benchmark. "
            "Memory covers current model and evaluation arrays, excluding "
            "Python overhead, logs, inputs and temporary allocations. "
            "Log event count is reported separately. No tail flush."
        ),
        "invalid_input_checks": (
            "One Inf score rejected without logging; one Inf fit rejected "
            "and logged. Model, evaluation counters/metrics and probe "
            "predictions unchanged; valid processing then continues."
        ),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "benchmark/results" / f"stream_long_run_{stamp}"
    output.mkdir(parents=True, exist_ok=False)

    with (output / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": metadata,
                "checkpoints": checkpoints,
                "final_metrics": trainer.metrics_result(),
                "final_log_events": len(logs),
                "failed_fit_events": len(failed),
                "checks_passed": True,
            },
            handle, default=json_default, indent=2, allow_nan=False,
        )
        handle.write("\n")

    print("\nLong-stream and invalid-input recovery checks: PASS")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()