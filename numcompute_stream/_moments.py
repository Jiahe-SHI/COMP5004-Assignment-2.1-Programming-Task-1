"""Vectorised, NaN-aware population moments shared by streaming components."""

import numpy as np


def merge_moments(X, previous=None):
    """Return independent (count, mean, M2) arrays for validated numeric X.

    X must be a nonempty two-dimensional real numeric array without infinity.
    Axis 0 contains samples. NaN values do not contribute to statistics.

    previous is None or a compatible (count, mean, M2) tuple.
    No raw history is retained. Arithmetic uses float64; unrepresentable
    intermediates raise FloatingPointError without changing previous arrays.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = X.astype(np.float64, copy=True)
        valid = ~np.isnan(values)
        count = valid.sum(axis=0, dtype=np.int64)
        observed = count > 0

        # Shift by an observed value for more stable chunk calculations.
        anchor = values[np.argmax(valid, axis=0), np.arange(X.shape[1])]
        anchor = np.where(observed, anchor, 0.0)
        shifted = np.where(valid, values, anchor) - anchor

        mean = anchor + np.sum(shifted / np.maximum(count, 1), axis=0)
        residual = np.where(valid, values, mean) - mean
        m2 = np.sum(residual * residual, axis=0)
        mean[~observed] = np.nan

        if previous is None:
            return count, mean, m2

        old_count, old_mean, old_m2 = previous
        if np.any(old_count > np.iinfo(np.int64).max - count):
            raise OverflowError("Per-feature observation count exceeds int64.")

        total = old_count + count
        merged_mean = old_mean.copy()
        merged_m2 = old_m2.copy()

        # Previously missing columns acquire their first valid statistics.
        first = (old_count == 0) & observed
        both = (old_count > 0) & observed
        merged_mean[first] = mean[first]
        merged_m2[first] = m2[first]

        # Merge columns observed in both the old state and the new chunk.
        delta = mean[both] - old_mean[both]
        weight = count[both] / total[both]
        merged_mean[both] = old_mean[both] + delta * weight
        correction = delta * np.sqrt(old_count[both] * weight)
        merged_m2[both] = (old_m2[both] + m2[both] + correction * correction)

        return total, merged_mean, merged_m2