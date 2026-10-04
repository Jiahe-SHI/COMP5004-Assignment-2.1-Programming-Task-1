"""Descriptive statistics with explicit axis and NaN handling."""

import numpy as np

from .utils import _as_float_array, _validate_axis, _validate_integer


def _validate_nan_policy(nan_policy):
    """Validate and return the NaN handling policy."""
    if not isinstance(nan_policy, str):
        raise TypeError("nan_policy must be a string.")

    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError(
            "nan_policy must be 'propagate', 'omit', or 'raise'."
        )

    return nan_policy


def _prepare_reduction(X, axis, nan_policy):
    """Validate statistical input and normalize the reduction axis."""
    nan_policy = _validate_nan_policy(nan_policy)

    array = _as_float_array(X, name="X")

    if np.any(np.isinf(array)):
        raise ValueError("X must not contain positive or negative infinity.")

    if axis is not None:
        axis = _validate_axis(axis, array.ndim)

    nan_mask = np.isnan(array)

    if nan_policy == "raise" and np.any(nan_mask):
        raise ValueError(
            "X must not contain NaN when nan_policy='raise'."
        )

    return array, axis, nan_policy, nan_mask


def _scalar_or_array(result):
    """Return a NumPy scalar when the result has no dimensions."""
    result = np.asarray(result, dtype=np.float64)

    if result.ndim == 0:
        return result[()]

    return result


def mean(X, axis=None, nan_policy="propagate"):
    """Return the arithmetic mean over all values or one axis."""
    array, axis, nan_policy, nan_mask = _prepare_reduction(
        X, axis, nan_policy
    )

    valid = ~nan_mask
    count = np.sum(valid, axis=axis)

    # Replacing NaN with zero makes summation safe for missing values.
    clean = np.where(valid, array, 0.0)
    total = np.sum(clean, axis=axis, dtype=np.float64)

    result = np.full(np.shape(total), np.nan, dtype=np.float64)

    # Empty or all-NaN slices remain NaN without division by zero.
    np.divide(total, count, out=result, where=count > 0)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)

def variance(X, axis=None, ddof=0, nan_policy="propagate"):
    """Return variance over all values or one axis."""
    ddof = _validate_integer(ddof, name="ddof", minimum=0)

    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    valid = ~nan_mask

    # Retain reduced dimensions so the mean broadcasts against X.
    count_keepdims = np.sum(valid, axis=axis, keepdims=True)

    clean = np.where(valid, array, 0.0)
    total = np.sum(clean,axis=axis,dtype=np.float64,keepdims=True,)

    center = np.zeros_like(total, dtype=np.float64)
    np.divide(total,count_keepdims,out=center, where=count_keepdims > 0,)

    # Missing positions contribute zero to the squared deviations.
    deviations = np.zeros_like(array, dtype=np.float64)
    np.subtract(array, center, out=deviations, where=valid)

    squared_deviations = deviations * deviations
    squared_sum = np.sum(squared_deviations,axis=axis, dtype=np.float64,)

    count = np.sum(valid, axis=axis)
    enough_data = count > ddof

    # Use a harmless denominator for invalid slices.
    denominator = np.where(enough_data, count - ddof, 1)

    result = np.full(np.shape(squared_sum),np.nan,dtype=np.float64,)

    np.divide(squared_sum,denominator,out=result,where=enough_data,)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)


def std(X, axis=None, ddof=0, nan_policy="propagate"):
    """Return standard deviation over all values or one axis."""
    result = variance(X, axis=axis, ddof=ddof, nan_policy=nan_policy)

    return _scalar_or_array(np.sqrt(result))

def _extreme(X, axis, nan_policy, *, largest):
    """Compute a minimum or maximum after shared validation."""
    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    valid = ~nan_mask
    has_valid = np.any(valid, axis=axis)

    # The replacement cannot win against a valid finite value.
    fill = -np.inf if largest else np.inf
    clean = np.where(valid, array, fill)

    reducer = np.max if largest else np.min
    result = reducer(clean, axis=axis, initial=fill)

    # Empty and all-NaN slices return NaN instead of infinity.
    result = np.where(has_valid, result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)


def min(X, axis=None, nan_policy="propagate"):
    """Return the minimum over all values or one axis."""
    return _extreme(X, axis, nan_policy, largest=False)


def max(X, axis=None, nan_policy="propagate"):
    """Return the maximum over all values or one axis."""
    return _extreme(X, axis, nan_policy, largest=True)


def _safe_midpoint(lower, upper):
    """Average ordered finite endpoints without avoidable overflow."""
    lower, upper = np.broadcast_arrays(lower, upper)
    same_sign = np.signbit(lower) == np.signbit(upper)

    result = np.empty(lower.shape, dtype=np.float64)

    # Same-sign endpoints have a representable difference.
    result[same_sign] = (lower[same_sign] + (upper[same_sign] - lower[same_sign]) * 0.5)

    # Opposite-sign endpoints have a representable sum.
    opposite_sign = ~same_sign
    result[opposite_sign] = (lower[opposite_sign] + upper[opposite_sign]) * 0.5

    return result


def median(X, axis=None, nan_policy="propagate"):
    """Return the median over all values or one axis."""
    array, axis, nan_policy, nan_mask = _prepare_reduction(X, axis, nan_policy)

    # Put the reduction dimension last for uniform indexing.
    if axis is None:
        values = array.reshape(-1)
    else:
        values = np.moveaxis(array, axis, -1)

    output_shape = values.shape[:-1]

    if values.shape[-1] == 0:
        return _scalar_or_array(np.full(output_shape, np.nan, dtype=np.float64))

    # NumPy places NaN after finite values in ascending order.
    ordered = np.sort(values, axis=-1, kind="mergesort")
    count = np.sum(~np.isnan(values), axis=-1)

    lower_index = np.maximum((count - 1) // 2, 0)
    upper_index = count // 2

    lower = np.take_along_axis(ordered, lower_index[..., None], axis=-1)[..., 0]

    upper = np.take_along_axis(ordered, upper_index[..., None], axis=-1)[..., 0]

    has_valid = count > 0

    # Use finite placeholders before restoring missing results.
    lower = np.where(has_valid, lower, 0.0)
    upper = np.where(has_valid, upper, 0.0)

    result = _safe_midpoint(lower, upper)
    result = np.where(has_valid, result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis)
        result = np.where(affected, np.nan, result)

    return _scalar_or_array(result)

def _validate_quantile_parameters(q, method):
    """Validate quantile levels and interpolation method."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("linear", "lower", "higher", "midpoint"):
        raise ValueError("method must be 'linear', 'lower', 'higher', or 'midpoint'.")

    levels = _as_float_array(q, name="q", finite=True)

    if levels.ndim > 1:
        raise ValueError("q must be a scalar or a 1D array.")

    if np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("q must contain values between 0 and 1.")

    return levels, method


def _linear_interpolate(lower, upper, weight):
    """Interpolate ordered finite endpoints without avoidable overflow."""
    lower, upper, weight = np.broadcast_arrays(lower, upper, weight)

    same_sign = np.signbit(lower) == np.signbit(upper)
    result = np.empty(lower.shape, dtype=np.float64)

    # Same-sign endpoints have a representable difference.
    result[same_sign] = (lower[same_sign] + weight[same_sign] * (upper[same_sign] - lower[same_sign]))

    # Opposite-sign endpoints may have an overflowing difference.
    opposite_sign = ~same_sign
    result[opposite_sign] = ((1.0 - weight[opposite_sign]) * lower[opposite_sign] + weight[opposite_sign] * upper[opposite_sign])

    # Preserve exact endpoint values.
    result = np.where(weight == 0.0, lower, result)
    result = np.where(weight == 1.0, upper, result)

    return result


def quantile(values,q,axis=None, method="linear",nan_policy="propagate",):
    """Return quantiles over all values or one axis."""
    levels, method = _validate_quantile_parameters(q, method)

    array, axis, nan_policy, nan_mask = _prepare_reduction(values, axis, nan_policy)

    scalar_q = levels.ndim == 0
    levels = levels.reshape(-1)

    if axis is None:
        data = array.reshape(-1)
    else:
        data = np.moveaxis(array, axis, -1)

    output_shape = data.shape[:-1]
    slice_length = data.shape[-1]
    result_shape = (levels.size,) + output_shape

    if slice_length == 0:
        result = np.full(result_shape, np.nan, dtype=np.float64)
        if scalar_q:
            result = result[0]
        return _scalar_or_array(result)

    # Each row represents one reduction slice.
    rows = data.reshape(-1, slice_length)
    ordered = np.sort(rows, axis=1, kind="mergesort")
    count = np.sum(~np.isnan(rows), axis=1)

    # Empty slices temporarily use position zero.
    last_index = np.maximum(count - 1, 0)
    positions = levels[:, None] * last_index[None, :]

    lower_index = np.floor(positions).astype(np.intp)
    upper_index = np.ceil(positions).astype(np.intp)

    row_index = np.arange(rows.shape[0])[None, :]

    lower = ordered[row_index, lower_index]
    upper = ordered[row_index, upper_index]

    has_valid = count > 0

    # Avoid NaN arithmetic before restoring missing results.
    lower = np.where(has_valid[None, :], lower, 0.0)
    upper = np.where(has_valid[None, :], upper, 0.0)

    if method == "lower":
        result = lower
    elif method == "higher":
        result = upper
    elif method == "midpoint":
        result = _safe_midpoint(lower, upper)
    else:
        weight = positions - lower_index
        result = _linear_interpolate(lower, upper, weight)

    result = np.where(has_valid[None, :], result, np.nan)

    if nan_policy == "propagate":
        affected = np.any(nan_mask, axis=axis).reshape(-1)
        result = np.where(affected[None, :], np.nan, result)

    result = result.reshape(result_shape)

    if scalar_q:
        result = result[0]

    return _scalar_or_array(result)

def _histogram_edges(values, bins, range):
    """Validate or construct finite, strictly increasing bin edges."""
    if isinstance(bins, (int, np.integer, bool, np.bool_)):
        n_bins = _validate_integer(bins, name="bins", minimum=1)

        if range is not None:
            bounds = _as_float_array(range, name="range", finite=True)

            if bounds.shape != (2,):
                raise ValueError("range must contain exactly two finite values.")

            lower, upper = bounds

            if lower >= upper:
                raise ValueError("range must satisfy lower < upper.")

        elif values.size == 0:
            lower, upper = 0.0, 1.0

        else:
            lower = np.min(values)
            upper = np.max(values)

            if lower == upper:
                lower = lower - 0.5
                upper = upper + 0.5

        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower >= upper
        ):
            raise ValueError("Cannot construct distinct finite bin edges; "
                             "provide a suitable explicit range or edge array.")

        # Interpolate without forming a potentially overflowing
        # difference between opposite-sign extreme endpoints.
        weights = np.linspace(
            0.0, 1.0, n_bins + 1, dtype=np.float64
        )
        edges = _linear_interpolate(lower, upper, weights)

    else:
        edges = _as_float_array(bins, name="bins", finite=True)

        if edges.ndim == 0:
            raise TypeError("bins must be a positive integer or a 1D edge array.")

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Bin edges must be a 1D array with at least two values.")

        if range is not None:
            raise ValueError("range must be None when explicit bin edges are used.")

    # Compare neighbors directly rather than subtracting them.
    if (
        not np.all(np.isfinite(edges))
        or np.any(edges[1:] <= edges[:-1])
    ):
        raise ValueError("Bin edges must be finite and strictly increasing; "
                         "use fewer bins or a suitable explicit range or edge array.")

    return edges


def histogram(values, bins=10, range=None, nan_policy="omit"):
    """Count 1D numeric values in histogram bins."""
    nan_policy = _validate_nan_policy(nan_policy)

    if nan_policy == "propagate":
        raise ValueError("histogram supports only nan_policy='omit' or 'raise'.")

    array, _, _, nan_mask = _prepare_reduction(values, None, nan_policy)

    if array.ndim != 1:
        raise ValueError("values must be a 1D array.")

    clean = array[~nan_mask]
    edges = _histogram_edges(clean, bins, range)
    n_bins = edges.size - 1

    # Include the final right endpoint; ignore out-of-range values.
    inside = ((clean >= edges[0]) & (clean <= edges[-1]))
    selected = clean[inside]

    indices = np.searchsorted(edges, selected, side="right") - 1

    # The final edge belongs to the last bin.
    indices = np.minimum(indices, n_bins - 1)

    counts = np.bincount(indices, minlength=n_bins)

    return counts, edges

class StreamingStats:
    """Accumulate statistics from independent 1D numeric batches.
    Stores count, mean and M2 without retaining previous batches.
    Uses the batch-merge equivalent of Welford's algorithm."""

    def __init__(self, nan_policy="propagate"):
        """Initialize an empty accumulator."""
        self.nan_policy = _validate_nan_policy(nan_policy)
        self.reset()

    def reset(self):
        """Clear accumulated statistics and return self."""
        self.count = 0
        self.mean = np.float64(np.nan)
        self.M2 = np.float64(0.0)
        self._poisoned = False

        return self

    def update(self, values):
        """Merge a 1D numeric batch into the accumulated state."""
        array, _, policy, nan_mask = _prepare_reduction(values, None, self.nan_policy)

        if array.ndim != 1:
            raise ValueError("values must be a 1D array.")

        if array.size == 0:
            return self

        clean = array[~nan_mask]
        batch_count = int(clean.size)
        new_count = self.count + batch_count

        propagate_nan = (policy == "propagate" and np.any(nan_mask))

        if self._poisoned or propagate_nan:
            self.count = new_count
            self.mean = np.float64(np.nan)
            self.M2 = np.float64(np.nan)
            self._poisoned = True
            return self

        # Under "omit", a batch containing only NaN changes nothing.
        if batch_count == 0:
            return self

        try:
            with np.errstate(over="raise",invalid="raise",divide="raise",):
                batch_mean = np.mean(clean, dtype=np.float64)
                deviations = clean - batch_mean
                batch_M2 = np.sum(deviations * deviations,dtype=np.float64,)

                if self.count == 0:
                    new_mean = batch_mean
                    new_M2 = batch_M2

                else:
                    delta = batch_mean - self.mean
                    batch_weight = batch_count / new_count

                    new_mean = (self.mean + delta * batch_weight)

                    # Equivalent to:
                    # delta**2 * old_count * batch_count / new_count
                    # Scale before squaring to reduce avoidable overflow.
                    merge_weight = self.count * batch_weight
                    scaled_delta = delta * np.sqrt(merge_weight)
                    correction = scaled_delta * scaled_delta

                    new_M2 = self.M2 + batch_M2 + correction

        except FloatingPointError as exc:
            raise ValueError("The batch statistics exceed supported float64 limits.") from exc

        if not np.isfinite(new_mean) or not np.isfinite(new_M2):
            raise ValueError("The updated statistics must remain finite.")

        # Commit only after the complete calculation succeeds.
        self.count = new_count
        self.mean = np.float64(new_mean)
        self.M2 = np.float64(new_M2)

        return self

    def variance(self, ddof=0):
        """Return accumulated variance as a numpy.float64 scalar."""
        ddof = _validate_integer(ddof, name="ddof", minimum=0)

        if self._poisoned or self.count <= ddof:
            return np.float64(np.nan)

        return np.float64(self.M2 / (self.count - ddof))

    def std(self, ddof=0):
        """Return accumulated standard deviation as a numpy.float64."""
        return np.float64(np.sqrt(self.variance(ddof=ddof)))