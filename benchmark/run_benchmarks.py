"""Run correctness checks and performance comparisons."""
import csv
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.benchmarking import benchmark
from numcompute.utils import _validate_integer
from benchmark.baselines import (
    loop_mean, loop_standardize, loop_mse, full_sort_topk,
)
from numcompute.stats import mean
from numcompute.preprocessing import StandardScaler
from numcompute.metrics import mse
from numcompute.sort_search import topk

def check_equivalent(expected, actual, *, exact=False):
    """Check output structure, shapes and numerical agreement.

    Parameters:
        expected, actual: Numeric scalars, arrays, or tuples of arrays.
            Corresponding outputs must have identical shapes.
        exact (bool): Use exact comparison when True, such as Top-k
            values and indices. Otherwise use floating-point tolerance.

    Returns:
        None. Successful completion means the outputs agree.

    Raises:
        AssertionError: Output structures, shapes or values differ.
        TypeError: exact is not boolean or comparison types are unsupported.
        ValueError: Comparison inputs have an invalid array structure.

    Complexity:
        Time O(N) and additional space O(N), where N is the total
        number of compared elements. Inputs are not modified.
    """
    if not isinstance(exact, (bool, np.bool_)):
        raise TypeError("exact must be a boolean.")

    if isinstance(expected, tuple) != isinstance(actual, tuple):
        raise AssertionError("Output structures differ.")

    expected_parts = expected if isinstance(expected, tuple) else (expected,)
    actual_parts = actual if isinstance(actual, tuple) else (actual,)

    if len(expected_parts) != len(actual_parts):
        raise AssertionError("Output counts differ.")

    for wanted, received in zip(expected_parts, actual_parts):
        # Check shapes explicitly before comparing values.
        if np.shape(wanted) != np.shape(received):
            raise AssertionError("Output shapes differ.")

        if exact:
            assert_array_equal(received, wanted)
        else:
            assert_allclose(
                received, wanted,
                rtol=1e-10, atol=1e-12, equal_nan=False,
            )

def run_case(case_id, baseline, candidate, args=(),
             exact=False, warmup=3, repeats=10):
    """Check correctness and time two implementations.

    Parameters:
        case_id (str): Name identifying the experiment.
        baseline, candidate (callable): Functions being compared.
        args (tuple): Shared inputs; shapes depend on the functions.
        exact (bool): Whether outputs must match exactly.
        warmup (int): Untimed calls per implementation, default 3.
            Must be non-negative.
        repeats (int): Measured calls per implementation, default 10.
            Must be positive. Boolean counts are rejected.

    Returns:
        dict: Case ID, correctness status, two timing dictionaries
            and scalar speedup. Each timing dictionary contains
            repeats samples. Speedup is None for invalid median times.

    Raises:
        AssertionError: Outputs differ in structure, shape or value.
        TypeError: Functions, counts or comparison options are invalid.
        ValueError: Counts are outside their allowed ranges.
        Errors from either implementation propagate unchanged.

    Complexity:
        Time O((w+r+1)*(C+1) + N + r log r), where C is the combined
        call cost and N is the total number of output elements.
        Additional space O(N+r), excluding function workspace.
    """
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    # Correctness checks happen outside the timed intervals.
    expected = baseline(*args)
    actual = candidate(*args)
    check_equivalent(expected, actual, exact=exact)

    baseline_timing = benchmark(
        baseline, *args, warmup=warmup, repeats=repeats
    )
    candidate_timing = benchmark(
        candidate, *args, warmup=warmup, repeats=repeats
    )

    baseline_seconds = baseline_timing["median_seconds"]
    candidate_seconds = candidate_timing["median_seconds"]

    # Only calculate a ratio from positive, finite times.
    if (baseline_seconds > 0 and candidate_seconds > 0
            and np.isfinite(baseline_seconds)
            and np.isfinite(candidate_seconds)):
        speedup = baseline_seconds / candidate_seconds
    else:
        speedup = None

    return {
        "case_id": case_id,
        "equivalent": True,
        "baseline": baseline_timing,
        "candidate": candidate_timing,
        "speedup": speedup,
    }

def prepare_inputs(n=1000, seed=42):
    """Create reproducible float64 inputs for the benchmark cases.

    Parameters:
        n (int): Number of samples, default 1000.
            Must be at least 10 to support fixed Top-k with k=10.
        seed (int): Non-negative random seed, default 42.
            Boolean values are rejected for both parameters.

    Returns:
        dict: values, y_true and y_pred have shape (n,).
            features has shape (n, 10), with its first column constant.
            All arrays contain finite float64 values.

    Raises:
        TypeError: n or seed is not an integer.
        ValueError: n is less than 10 or seed is negative.

    Complexity:
        Time O(n) and additional space O(n).
        The feature count is fixed at 10.
    """
    n = _validate_integer(n, name="n", minimum=10)
    seed = _validate_integer(seed, name="seed", minimum=0)

    # Use a local generator so global random state is unchanged.
    rng = np.random.default_rng(seed)

    values = rng.normal(size=n)
    features = rng.normal(size=(n, 10))
    y_true = rng.normal(size=n)
    y_pred = rng.normal(size=n)

    # Include a constant feature to exercise the scaler's rule.
    features[:, 0] = 7.0

    return {
        "values": values,
        "features": features,
        "y_true": y_true,
        "y_pred": y_pred,
    }

def project_standardize(X):
    """Fit and transform data with a newly created StandardScaler.

    Parameters:
        X (array-like, shape (n, p)): Non-empty numeric table.
            NaN is allowed, but each column needs a valid training value.

    Returns:
        numpy.ndarray: Independent float64 output of shape (n, p).
            Input NaN positions are preserved.

    Raises:
        TypeError: Input contains unsupported numeric types.
        ValueError: Shape, feature values or numeric range are invalid.

    Complexity:
        Time O(n * p). Additional space O(n * p).
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def run_experiments(n=1000, seed=42, warmup=3, repeats=10):
    """Run the four benchmark tasks at one input size.

    Parameters:
        n (int): Sample count, at least 10; default 1000.
        seed (int): Non-negative random seed; default 42.
        warmup (int): Non-negative untimed calls; default 3.
        repeats (int): Positive measured calls; default 10.
            Boolean values are rejected for integer parameters.

    Returns:
        list: Five result dictionaries containing correctness status,
            raw timings, speedup and case metadata.
            Numeric inputs have shape (n,), except features (n, 10).

    Raises:
        TypeError: Integer parameters have unsupported types.
        ValueError: Parameters are outside their allowed ranges.
        AssertionError: Implementations produce different results.
        Underlying computation errors propagate unchanged.

    Complexity:
        Time O((w+r+1) * n log n + r log r), where w is warmup
        and r is repeats. Additional space O(n+r), for five cases
        and a fixed feature count of 10.
    """
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    # Generate all inputs before starting the timed calls.
    data = prepare_inputs(n, seed)
    values = data["values"]
    features = data["features"]
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    n = values.size

    # Each case contains its name, functions, inputs and comparison mode.
    cases = [
        ("mean", loop_mean, mean, (values,), False),
        ("standardization", loop_standardize, project_standardize,
         (features,), False),
        ("mse", loop_mse, mse, (y_true, y_pred), False),
        ("topk_fixed", full_sort_topk, topk, (values, 10), True),
        ("topk_fraction", full_sort_topk, topk,
         (values, n // 10), True),
    ]

    results = []

    for case in cases:
        name, baseline, candidate, args, exact = case

        result = run_case(
            name + "_" + str(n),
            baseline, candidate,
            args=args, exact=exact,
            warmup=warmup, repeats=repeats,
        )

        # Keep the settings needed to interpret and export results.
        result["task"] = name
        result["shape"] = list(args[0].shape)
        result["dtype"] = str(args[0].dtype)
        result["baseline_name"] = baseline.__name__
        result["candidate_name"] = candidate.__name__

        if exact:
            result["task"] = "topk"
            result["k"] = args[1]
            result["comparison"] = "full_sort_vs_argpartition"
        else:
            result["k"] = None
            result["comparison"] = "loop_vs_vectorized"

        results.append(result)

    return results

def save_raw_timings(results, path, run_id):
    """Write individual timing samples to a new CSV file.

    Parameters:
        results (list): Result dictionaries from run_experiments.
        path (str or Path): Destination CSV file.
        run_id (str): Non-empty identifier linking one experiment run.

    Returns:
        Path: Saved CSV location. The file has five columns and one
            data row per timing sample. Repeat indices start at 1.

    Raises:
        TypeError: run_id is not a string or path is unsupported.
        ValueError: run_id is empty or contains only whitespace.
        FileExistsError: The destination file already exists.
        KeyError: A required result field is missing.
        OSError: The directory or file cannot be created or written.

    Complexity:
        Time O(C + R), where C is the number of cases and R is the
        total number of timing samples. Additional space O(1),
        excluding the supplied results and file I/O buffers.
    """
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")

    if not run_id.strip():
        raise ValueError("run_id must not be empty.")

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new file so an existing result is not overwritten.
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "run_id", "case_id", "implementation",
            "repeat_index", "seconds_per_call",
        ])

        for result in results:
            for implementation in ("baseline", "candidate"):
                times = result[implementation]["times_seconds"]

                for repeat_index, seconds in enumerate(times, start=1):
                    writer.writerow([
                        run_id,
                        result["case_id"],
                        implementation,
                        repeat_index,
                        seconds,
                    ])

    return csv_path