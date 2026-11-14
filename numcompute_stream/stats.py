"""Incremental descriptive statistics for numeric feature matrices."""

import numpy as np

from ._moments import merge_moments
from ._validation import validate_numeric_X


class StreamingStats:
    """Accumulate numeric moments and exact global quantiles.

    X has shape (samples, features). NaN is excluded independently in
    each feature; infinity and nonnumeric inputs are rejected.

    update_stats accumulates observations and returns self.
    result returns independent count, mean and population variance arrays.
    quantile uses all valid observations received since the last reset.
    reset clears statistics, retained history and the feature schema.

    Unobserved columns have count 0 and mean/variance/quantiles NaN.
    Calculations and retained history use float64; large integers can
    round on conversion. Moment calculations retain their numerical limits.

    History memory grows with valid observations and retained chunk
    references. Quantile queries allocate temporary concatenated arrays.
    Optional fixed bin_edges enable per-feature cumulative histograms.
    All features share the same finite, strictly increasing edges.
    Histogram state uses O(n_features * n_bins) additional memory.
    reset retains the configured edges while clearing learned state.
    """

    def __init__(self, bin_edges=None):
        self._bin_edges = self._validate_bin_edges(bin_edges)
        self.reset()

    def reset(self):
        """Clear learning state while retaining histogram configuration."""
        self._moments = None
        self._history = None
        self._hist_counts = None
        self.n_features_in_ = None
        self.n_rows_seen_ = 0
        return self

    def update_stats(self, X_chunk):
        """Update moments, retained history and optional histogram counts."""
        checked = validate_numeric_X(
            X_chunk, allow_nan=True, n_features=self.n_features_in_,
        )

        candidate = merge_moments(checked, self._moments)
        n_rows = self.n_rows_seen_ + checked.shape[0]

        with np.errstate(over="raise", invalid="raise"):
            values = checked.astype(np.float64, copy=True)

        hist_counts = None
        if self._bin_edges is not None:
            if self._hist_counts is None:
                hist_counts = np.zeros(
                    (checked.shape[1], self._bin_edges.size - 1),
                    dtype=np.int64,
                )
            else:
                hist_counts = self._hist_counts.copy()

        history = []
        for feature in range(checked.shape[1]):
            column = values[:, feature]
            observed = column[~np.isnan(column)]

            previous = (
                [] if self._history is None
                else self._history[feature]
            )
            chunks = list(previous)
            if observed.size:
                chunks.append(observed)
            history.append(chunks)

            if hist_counts is not None:
                counts, _ = np.histogram(observed, bins=self._bin_edges)
                counts = counts.astype(np.int64, copy=False)

                limit = np.iinfo(np.int64).max
                if np.any(hist_counts[feature] > limit - counts):
                    raise OverflowError("Histogram count exceeds int64.")
                hist_counts[feature] += counts

        # Publish only after all candidate state has been prepared.
        self._moments = candidate
        self._history = history
        self._hist_counts = hist_counts
        self.n_features_in_ = checked.shape[1]
        self.n_rows_seen_ = n_rows
        return self

    def result(self):
        """Return independent per-feature count, mean and variance arrays.

        n_rows includes entirely missing rows. Each array has shape
        (n_features,). Requires at least one successful update.
        """
        if self._moments is None:
            raise RuntimeError("StreamingStats requires an update before result.")

        count, mean, m2 = self._moments
        variance = np.full(mean.shape, np.nan)
        np.divide(m2, count, out=variance, where=count > 0)

        return {
            "n_rows": self.n_rows_seen_,
            "count": count.copy(),
            "mean": mean.copy(),
            "var": variance,
        }

    def quantile(self, q):
        """Return per-feature global quantiles using linear interpolation.

        q must be a real scalar or a nonempty one-dimensional numeric
        sequence with finite values in [0, 1].

        Scalar q returns shape (n_features,).
        A sequence of k probabilities returns shape (k, n_features).
        Unobserved columns return NaN without all-NaN reduction warnings.
        Querying does not modify accumulated state.
        """
        if self._moments is None:
            raise RuntimeError("StreamingStats requires an update before quantile.")

        probabilities = np.asarray(q)
        if probabilities.ndim > 1:
            raise ValueError("q must be a scalar or one-dimensional.")
        if probabilities.dtype.kind not in "iuf":
            raise TypeError("q must contain real numeric probabilities.")
        if probabilities.size == 0:
            raise ValueError("q must not be empty.")
        if not np.isfinite(probabilities).all():
            raise ValueError("q must contain only finite probabilities.")
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("q must lie within [0, 1].")

        scalar = probabilities.ndim == 0
        probabilities = probabilities.astype(np.float64).reshape(-1)

        output = np.full((probabilities.size, self.n_features_in_), np.nan, dtype=np.float64,)

        for feature, chunks in enumerate(self._history):
            if not chunks:
                continue

            observed = np.concatenate(chunks)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                output[:, feature] = np.quantile(observed, probabilities)

        return output[0] if scalar else output

    @staticmethod
    def _validate_bin_edges(bin_edges):
        """Return an independent float64 edge array, or None."""
        if bin_edges is None:
            return None

        edges = np.asarray(bin_edges)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be one-dimensional with at least two edges.")
        if edges.dtype.kind not in "iuf":
            raise TypeError("bin_edges must contain real numeric values.")
        if not np.isfinite(edges).all():
            raise ValueError("bin_edges must contain only finite values.")

        with np.errstate(over="raise", invalid="raise"):
            edges = edges.astype(np.float64, copy=True)

        if not np.isfinite(edges).all():
            raise ValueError("bin_edges must be representable as float64.")
        if np.any(edges[1:] <= edges[:-1]):
            raise ValueError("bin_edges must be strictly increasing after float64 conversion.")

        return edges

    def histogram(self):
        """Return independent histogram count and edge arrays.

        counts has shape (n_features, n_bins).
        bin_edges has shape (n_bins + 1,).

        Bins are left-closed and right-open, except the last bin, which
        includes its right endpoint. NaN and out-of-range values are
        excluded from histogram counts only.
        """
        if self._bin_edges is None:
            raise ValueError("Histogram requires bin_edges at construction.")
        if self._moments is None:
            raise RuntimeError("StreamingStats requires an update before histogram.")

        return {"counts": self._hist_counts.copy(),
                "bin_edges": self._bin_edges.copy(),}