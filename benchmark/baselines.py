from numcompute.utils import _as_float_array, _validate_integer
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

def loop_mse(y_true, y_pred):
    """Compute mean squared error using a Python loop.

    Parameters:
        y_true, y_pred (array-like): Finite real numeric data with
            identical shape (n,) or (n, p). Inputs must be non-empty.

    Returns:
        float: Scalar mean squared error over all elements.

    Raises:
        TypeError: Inputs contain unsupported numeric types.
        ValueError: Inputs have invalid dimensions, unequal shapes,
            no elements, irregular shapes, or non-finite values.

    Complexity:
        Time O(N). Additional space O(N) for validated copies,
        where N is the total number of elements in either input.

    Notes:
        Inputs are not modified. Intended for ordinary benchmark
        data; extreme magnitudes may overflow intermediate calculations.
    """
    # Validate finite numeric inputs and create independent float64 copies.
    true = _as_float_array(y_true, name="y_true", finite=True)
    pred = _as_float_array(y_pred, name="y_pred", finite=True)

    if true.ndim not in (1, 2) or pred.ndim not in (1, 2):
        raise ValueError("Inputs must be 1D or 2D.")
    # require exact shape matching; broadcast is banned in this part
    if true.shape != pred.shape:
        raise ValueError("Inputs must have identical shapes.")

    if true.size == 0:
        raise ValueError("Inputs must not be empty.")
    # accumulate aquared errors over all elements using python floats
    total = 0.0
    for actual, predicted in zip(true.flat, pred.flat):
        difference = float(actual) - float(predicted)
        total += difference * difference
    
    #divide by total element count if the inputs is 2d
    return total / true.size

def full_sort_topk(values, k):
    """Select the largest k values using a complete sort.

    Parameters:
        values (array-like, shape (n,)): Finite real numeric data,
            converted to float64. Benchmark inputs use float64.
        k (int): Number of selected values, satisfying 0 <= k <= n.
            Boolean values are rejected.

    Returns:
        tuple: Selected float64 values and original integer indices,
            both with shape (k,). Values are descending; ties use
            ascending original indices. k=0 returns empty arrays.

    Raises:
        TypeError: Input has unsupported types or k is not an integer.
        ValueError: Input is not 1D, contains non-finite values,
            has an irregular shape, or k is outside [0, n].

    Complexity:
        Time O(n log n). Additional space O(n).

    Notes:
        Input is not modified. The baseline sorts every element.
        Equivalence with project topk is checked on float64 data.
    """
    data = _as_float_array(values, ndim=1, finite=True)
    k = _validate_integer(k, name="k", minimum=0)

    if k > data.size:
        raise ValueError("k must not exceed the number of values.")

    # Original indices provide the tie-breaking key.
    indices = np.arange(data.size)

    # Negative values give descending order; indices break ties.
    order = np.lexsort((indices, -data))

    # Select only after completing the full sort.
    selected = order[:k]
    return data[selected], selected