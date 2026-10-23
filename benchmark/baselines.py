from numcompute.utils import _as_float_array
import math
import numpy as np

def loop_mean(values):
    """Compute a mean using a Python loop.

    Parameters:
        values (array-like, shape (n,)): Non-empty finite real data.
            Converted to an independent float64 array.

    Returns:
        float: Scalar arithmetic mean over all n values.

    Raises:
        TypeError: Input contains unsupported types, including
            booleans, strings, complex values or object arrays.
        ValueError: Input is not 1D, is empty, has an irregular
            shape, or contains NaN or infinity.

    Complexity:
        Time O(n). Additional space O(n) for the validated copy.

    Notes:
        The input is not modified. This reference targets ordinary
        benchmark data; extreme magnitudes may overflow during summation.
    """
    data = _as_float_array(values, ndim=1, finite=True)
    
    if data.size == 0: # check input and create an independent float
        raise ValueError("values must not be empty.")

    total = 0.0
    for value in data: # commulative summation
        total += float(value)

    return total / data.size # gain mean by deviation of quantity

def loop_standardize(X):
    """Standardize feature columns using Python loops.

    Parameters:
        X (array-like, shape (n, p)): Finite real numeric data.
            Both sample count n and feature count p must be positive.

    Returns:
        numpy.ndarray: Independent float64 array of shape (n, p).
            Columns use population standard deviation, ddof=0.
            Constant columns use a denominator of 1 and become zero.

    Raises:
        TypeError: Input contains unsupported types, including
            booleans, strings, complex values or object arrays.
        ValueError: Input is not 2D, has an empty dimension,
            has an irregular shape, or contains NaN or infinity.

    Complexity:
        Time O(n * p). Additional space O(n * p), including
        the validated input copy and output array.

    Notes:
        Computes column statistics and transforms data in one call.
        The input is not modified. Intended for ordinary benchmark
        data; extreme magnitudes may overflow intermediate calculations.
    """
    # validate 2D input and create an in dependent float
    data = _as_float_array(X, ndim=2, finite=True)
    rows, columns = data.shape

    if rows == 0 or columns == 0:
        raise ValueError("X must have samples and features.")

    result = np.empty_like(data)

    for column in range(columns):
        # Caculate the mean of the current feature column
        total = 0.0
        for row in range(rows):
            total += float(data[row, column])
        mean = total / rows

        # Population variance uses ddof=0.
        squared_total = 0.0
        for row in range(rows):
            difference = float(data[row, column]) - mean
            squared_total += difference * difference

        # Constant columns use a denominator of 1.
        scale = math.sqrt(squared_total / rows) or 1.0

        for row in range(rows):
            result[row, column] = (
                float(data[row, column]) - mean
            ) / scale

    return result