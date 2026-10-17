"""Finite-difference gradients and Jacobians with explicit validation."""

from numbers import Real
import numpy as np
from .utils import _as_float_array


def _validate_method(method):
    """Validate and return the finite-difference method name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'.")

    return method


def _validate_step(h):
    """Validate h and return it as a positive, finite Python float."""
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, Real):
        raise TypeError("h must be a real numeric scalar.")

    try:
        value = float(h)
    except (OverflowError, ValueError) as exc:
        raise ValueError("h must be representable as finite float64.") from exc

    if not np.isfinite(value):
        raise ValueError("h must be finite.")

    if value <= 0.0:
        raise ValueError(f"h must be positive; got {value}.")

    return value



def _prepare(x, method, h):
    """Validate inputs and return x, the method and the perturbed points.

    Return (array, method, upper, lower). For the central method, lower
    holds x - h; for the forward method, lower is the unperturbed x.
    """
    method = _validate_method(method)
    h = _validate_step(h)
    array = _as_float_array(x, name="x", ndim=1, finite=True)

    if array.size == 0:
        raise ValueError("x must contain at least one value.")

    # Overflow is reported below as a non-finite coordinate.
    with np.errstate(over="ignore"):
        upper = array + h
        lower = array - h if method == "central" else array.copy()

    for point in (upper, lower) if method == "central" else (upper,):
        if not np.all(np.isfinite(point)):
            raise ValueError("x + h and x - h must remain finite.")

        # A step below the spacing of float64 values near x has no effect.
        if np.any(point == array):
            raise ValueError(
                "h is too small to change every coordinate of x; "
                "use a larger h."
            )

    with np.errstate(over="ignore"):
        steps = upper - lower

    if not np.all(np.isfinite(steps)):
        raise ValueError("The step between perturbed points must be finite.")

    return array, method, upper, lower


def _perturbed(array, index, value):
    """Return an independent copy of array with one coordinate replaced."""
    point = array.copy()
    point[index] = value
    return point


def _scalar_output(value):
    """Validate one output of f for grad and return it as a float."""
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 0:
        raise ValueError(
            f"f(x) must return a scalar; got shape {result.shape}."
        )

    if not np.isfinite(result):
        raise ValueError("f(x) must return a finite value.")

    return float(result)


def _vector_output(value, n_outputs=None):
    """Validate one output of f for jacobian and return a float64 array."""
    result = _as_float_array(value, name="f(x)")

    if result.ndim != 1:
        raise ValueError(
            f"f(x) must return a 1D array; got {result.ndim}D."
        )

    if result.size == 0:
        raise ValueError("f(x) must return at least one value.")

    if n_outputs is not None and result.size != n_outputs:
        raise ValueError(
            f"f(x) must return {n_outputs} values on every evaluation; "
            f"got {result.size}."
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("f(x) must return only finite values.")

    return result
