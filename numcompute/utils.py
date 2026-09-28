"""Shared validation and numerical utilities."""

from numbers import Real
import numpy as np


def _as_float_array(x, *, name="x", ndim=None, finite=False):
    """Validate real numeric input and return an independent float64 array.

    Parameters
    ----------
    x : array-like
        Real numeric input. Boolean, complex, string and object arrays
        are not supported.
    name : str
        Input name used in error messages.
    ndim : int or None
        Required number of dimensions, or None for any shape.
    finite : bool
        If True, reject NaN and positive/negative infinity.

    Returns
    -------
    numpy.ndarray
        Independent float64 array preserving the input shape.

    Raises
    ------
    TypeError
        If the input contains unsupported types.
    ValueError
        If the shape is invalid or finite=True and values are non-finite.

    Notes
    -----
    Time: O(n). Additional space: O(n), where n is the element count.
    """

    try:
        array = np.asarray(x)
    except ValueError as exc:
        raise ValueError(
            f"{name} must have a regular rectangular shape."
        ) from exc

    # Check Python sequences before NumPy can silently convert mixed
    # values such as [True, 2] into an integer array.
    if isinstance(x, (list, tuple)):
        try:
            elements = np.asarray(x, dtype=object)
        except ValueError as exc:
            raise ValueError(
                f"{name} must have a regular rectangular shape."
            ) from exc

        for value in elements.flat:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Real
            ):
                raise TypeError(
                    f"{name} must contain only real numeric values."
                )


    # i: signed integer; u: unsigned integer; f: floating point.
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"{name} must contain real numbers, not boolean, "
            "complex, string or object data."
        )

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D; got {array.ndim}D."
        )

    result = array.astype(np.float64, copy=True)

    if finite and not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return result


def _validate_integer(value, *, name, minimum=None):
    """Validate a Python/NumPy integer and return a Python int.

    Boolean values are rejected. If minimum is supplied, the value
    must be at least minimum.

    Raises TypeError for unsupported types and ValueError for values
    below minimum.

    Time: O(1). Additional space: O(1), for fixed-width integers.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")

    value = int(value)

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}; got {value}."
        )

    return value