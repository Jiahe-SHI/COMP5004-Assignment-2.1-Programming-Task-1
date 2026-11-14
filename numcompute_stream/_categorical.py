"""Validation and batch factorisation for categorical feature columns."""

from numbers import Integral, Real
import numpy as np
from ._validation import validate_integer


def validate_categorical_X(X, *, allow_empty=False, n_features=None):
    """Return validated columns as (values, missing_mask, kind) tuples.

    Input shape is (samples, features). Each column contains strings or
    real numeric categories, optionally with None/NaN missing values.
    kind is 'string', 'numeric', or None for an entirely missing column.

    Object values require scalar type inspection. Category factorisation
    is performed separately using NumPy, not a Python sample-encoding loop.
    Returned column arrays are independent of caller-owned input.
    """
    # Preserve heterogeneous lists rather than coercing numbers to strings.
    if isinstance(X, np.ndarray):
        array = np.asarray(X)
    else:
        array = np.asarray(X, dtype=object)

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

    if array.dtype.kind not in "iufUO":
        raise TypeError("X must contain strings or real numeric categories.")

    columns = []
    for feature in range(array.shape[1]):
        column = array[:, feature]

        if column.dtype.kind == "U":
            values = column.astype(object)
            missing = np.zeros(column.size, dtype=bool)
            kind = "string"

        elif column.dtype.kind in "iuf":
            if np.isinf(column).any():
                raise ValueError(f"Feature {feature} must not contain infinity.")
            missing = np.isnan(column)
            # Object storage preserves integer values without float coercion.
            values = column.astype(object)
            kind = "numeric" if np.any(~missing) else None

        else:
            values, missing, kind = _validate_object_column(column, feature)

        values[missing] = None
        columns.append((values, missing, kind))

    return columns


def _validate_object_column(column, feature):
    """Inspect object types and normalise supported numeric scalars."""
    values = np.empty(column.size, dtype=object)
    missing = np.zeros(column.size, dtype=bool)
    kinds = set()

    for index, value in enumerate(column):
        if value is None:
            missing[index] = True
            values[index] = None
            continue

        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"Feature {feature} must not contain boolean values.")

        if isinstance(value, str):
            values[index] = str(value)
            kinds.add("string")

        elif isinstance(value, Integral):
            values[index] = int(value)
            kinds.add("numeric")

        elif isinstance(value, Real):
            number = float(value)
            if np.isnan(number):
                missing[index] = True
                values[index] = None
            elif not np.isfinite(number):
                raise ValueError(f"Feature {feature} must not contain infinity.")
            else:
                values[index] = number
                kinds.add("numeric")

        else:
            raise TypeError(f"Feature {feature} contains an unsupported category type.")

    if len(kinds) > 1:
        raise TypeError(f"Feature {feature} must not mix strings and numbers.")

    kind = next(iter(kinds)) if kinds else None
    return values, missing, kind


def factorise_column(values, missing):
    """Return sorted ordinary categories and per-row local codes.

    Inputs must come from validate_categorical_X.
    Missing rows receive code -1. Missing is not included in categories.
    Codes describe this chunk only, not a persistent encoder mapping.
    """
    categories, inverse = np.unique(values[~missing], return_inverse=True)
    codes = np.full(values.shape, -1, dtype=np.intp)
    codes[~missing] = inverse
    return categories, codes