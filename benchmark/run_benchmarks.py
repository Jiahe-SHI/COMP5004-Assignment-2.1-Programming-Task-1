"""Run correctness checks and performance comparisons."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


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