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
