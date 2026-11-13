"""Shared validation for streaming estimators.

Validation does not change estimator state or caller-owned arrays.
Numeric features use axis 0 for samples and axis 1 for features.
"""

from numbers import Integral

import numpy as np


def validate_integer(value, name, *, minimum=0):
    """Return a Python integer, rejecting booleans and non-integers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def validate_numeric_X(X, *, allow_nan=False, allow_empty=False, n_features=None):
    """Validate a real numeric matrix with shape (samples, features).

    Integer and floating dtypes are accepted without conversion to float.
    NaN is allowed only when requested; infinity is always rejected.
    Numeric strings, booleans, complex values and object arrays are rejected.

    The returned array may share memory with X. Components retaining input
    data must make their own copy.
    """
    array = np.asarray(X)

    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    if n_features is not None:
        expected = validate_integer(n_features, "n_features", minimum=1)
        if array.shape[1] != expected:
            raise ValueError(
                f"X must contain {expected} features; "
                f"got {array.shape[1]}."
            )

    if array.dtype.kind not in "iuf":
        raise TypeError("X must contain real integer or floating values.")
    if np.isinf(array).any():
        raise ValueError("X must not contain infinity.")
    if not allow_nan and np.isnan(array).any():
        raise ValueError("X must not contain NaN.")

    return array


def validate_y(y, *, n_samples=None, allow_empty=False):
    """Validate homogeneous integer or Unicode string labels.

    Reject boolean, floating, mixed-type and missing labels.
    Preserve integer values without conversion through float.

    Lists and tuples are first represented as objects to avoid coercing
    mixed labels, such as [1, "cat"], into strings. Already constructed
    arrays are validated according to their current values and dtype.
    """
    if isinstance(y, np.ndarray):
        array = np.asarray(y)
    else:
        array = np.asarray(y, dtype=object)

    if array.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if not allow_empty and array.size == 0:
        raise ValueError("y must contain at least one sample.")

    if n_samples is not None:
        expected = validate_integer(n_samples, "n_samples", minimum=0)
        if array.size != expected:
            raise ValueError(
                f"X and y must contain the same number of samples; "
                f"expected {expected}, got {array.size}."
            )

    # An empty object array has no label type to infer.
    if array.size == 0 and array.dtype.kind == "O":
        return array

    if array.dtype.kind in "iuU":
        return array

    if array.dtype.kind == "O":
        # Object inputs require type inspection. Normal typed NumPy arrays
        # take the dtype-based path above without a per-sample Python loop.
        integers = all(isinstance(value, Integral)
                       and not isinstance(value, (bool, np.bool_)) for value in array)
        strings = all(isinstance(value, str) for value in array)
        if integers or strings:
            return array

    raise TypeError(
        "y must contain homogeneous integers or homogeneous strings; "
        "boolean, floating, mixed-type and missing labels are unsupported."
    )


def validate_X_y(X, y, *, allow_nan=False):
    """Validate nonempty training inputs without modifying either input."""
    X_checked = validate_numeric_X(X, allow_nan=allow_nan)
    y_checked = validate_y(y, n_samples=X_checked.shape[0])
    return X_checked, y_checked


def validate_tree_parameters(
        *, max_depth, min_samples_split, max_features,
        buffer_size, criterion, random_state, n_features=None,
):
    """Validate tree configuration before any learning state changes.

    max_depth is None or a nonnegative integer.
    max_features is None, a positive integer, 'sqrt', or 'log2'.
    random_state is None or a nonnegative integer seed.
    An integer max_features must not exceed the known feature count.
    """
    if max_depth is not None:
        validate_integer(max_depth, "max_depth", minimum=0)

    validate_integer(min_samples_split, "min_samples_split", minimum=2)
    validate_integer(buffer_size, "buffer_size", minimum=1)

    if n_features is not None:
        n_features = validate_integer(n_features, "n_features", minimum=1)

    if max_features is not None:
        if isinstance(max_features, str):
            if max_features not in ("sqrt", "log2"):
                raise ValueError(
                    "max_features must be None, a positive integer, "
                    "'sqrt', or 'log2'."
                )
        else:
            count = validate_integer(max_features, "max_features", minimum=1)
            if n_features is not None and count > n_features:
                raise ValueError(
                    "max_features must not exceed the number of features."
                )

    if not isinstance(criterion, str):
        raise TypeError("criterion must be a string.")
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'.")

    if random_state is not None:
        validate_integer(random_state, "random_state", minimum=0)