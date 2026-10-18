import numpy as np

from .sort_search import _as_numeric_array
from .stats import _validate_nan_policy, quantile
from .utils import _as_float_array


def rank(values, method="average", nan_policy="propagate"):
    array = _as_numeric_array(values)
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if method not in ("average", "dense", "ordinal"):
        raise ValueError("method must be 'average', 'dense', or 'ordinal'.")
    nan_policy = _validate_nan_policy(nan_policy)
    missing = np.isnan(array)
    if nan_policy == "raise" and np.any(missing):
        raise ValueError("values must not contain NaN when nan_policy='raise'.")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    if (nan_policy == "propagate" and np.any(missing)) or array.size == 0:
        return result
    valid_indices = np.flatnonzero(~missing)
    data = array[valid_indices]
    if data.size == 0:
        return result
    order = np.argsort(data, kind="stable")
    ordered = data[order]
    if method == "ordinal":
        sorted_ranks = np.arange(1, data.size + 1, dtype=np.float64)
    else:
        starts = np.r_[True, ordered[1:] != ordered[:-1]]
        group = np.cumsum(starts) - 1
        if method == "dense":
            sorted_ranks = (group + 1).astype(np.float64)
        else:
            first = np.flatnonzero(starts)
            end = np.r_[first[1:], data.size]
            sorted_ranks = ((first + 1.0 + end) / 2.0)[group]
    result[valid_indices[order]] = sorted_ranks
    return result


def percentile(values, q, method="linear", axis=None,
               nan_policy="propagate", *, interpolation=None):
    if interpolation is not None:
        if not isinstance(interpolation, str):
            raise TypeError("interpolation must be a string.")
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        if method != "linear" and method != interpolation:
            raise ValueError("method and interpolation must agree.")
        method = interpolation
    levels = _as_float_array(q, name="q", finite=True)
    if levels.ndim > 1:
        raise ValueError("q must be a scalar or a 1D array.")
    if np.any((levels < 0) | (levels > 100)):
        raise ValueError("q must contain values between 0 and 100.")
    return quantile(values, levels / 100.0, axis=axis,
                    method=method, nan_policy=nan_policy)
