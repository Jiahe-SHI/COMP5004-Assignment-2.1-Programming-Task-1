"""Run correctness checks and performance comparisons."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.benchmarking import benchmark
from numcompute.utils import _validate_integer

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