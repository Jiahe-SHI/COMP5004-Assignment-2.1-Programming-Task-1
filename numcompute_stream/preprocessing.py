"""Incremental numeric preprocessing; samples occupy axis 0."""

import numpy as np
from ._moments import merge_moments
from ._validation import validate_numeric_X
from ._categorical import validate_categorical_X, factorise_column


class StandardScaler:
    """Standardise using cumulative per-feature population moments (ddof=0).

    fit resets statistics; partial_fit accumulates all observed values.
    NaN is ignored while learning and preserved during transformation.
    Infinity and nonnumeric inputs are rejected. Training requires at least
    one row and feature; fitted transforms accept zero rows.

    n_samples_seen_ is a per-feature int64 count. mean_, var_, scale_ are
    float64 vectors. Unobserved columns have mean/variance NaN and scale 1;
    their transformed values are NaN until learning observes that column.
    Constant columns use scale 1. Methods never modify caller-owned inputs.

    Persistent memory is O(n_features), temporary memory O(chunk size).
    Float64 conversion can round large integers. Unrepresentable arithmetic
    raises FloatingPointError; failed learning preserves the previous state.
    """

    def fit(self, X, y=None):
        """Replace learning state from nonempty X; y is accepted and ignored."""
        return self._learn(X, reset=True)

    def partial_fit(self, X, y=None):
        """Merge a nonempty chunk without retaining raw observations."""
        return self._learn(X, reset=False)

    def _learn(self, X, *, reset):
        fitted = hasattr(self, "n_features_in_")
        expected = self.n_features_in_ if fitted and not reset else None
        checked = validate_numeric_X(X, allow_nan=True, n_features=expected)

        previous = None
        if fitted and not reset:
            previous = (self.n_samples_seen_, self.mean_, self._m2)

        count, mean, m2 = merge_moments(checked, previous)

        variance = np.full(mean.shape, np.nan)
        np.divide(m2, count, out=variance, where=count > 0)

        scale = np.ones(mean.shape)
        np.sqrt(variance, out=scale, where=(count > 0) & (variance > 0),)

        # Publish learned state only after successful calculation.
        self.n_features_in_ = checked.shape[1]
        self.n_samples_seen_ = count
        self.mean_ = mean
        self._m2 = m2
        self.var_ = variance
        self.scale_ = scale
        return self

    def transform(self, X):
        """Return a new float64 matrix without changing learned statistics."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("StandardScaler must be fitted before transform.")

        checked = validate_numeric_X(X, allow_nan=True, allow_empty=True,
            n_features=self.n_features_in_,)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return (checked.astype(np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        """Fit from scratch and transform the same input."""
        return self.fit(X, y).transform(X)

class Imputer:
    """Incrementally impute numeric NaN values along axis 0.

    strategy='mean' uses cumulative observed means, with fill_value as
    the fallback for unobserved columns. strategy='constant' always uses
    fill_value. Infinity and nonnumeric inputs are rejected.

    fit resets learning; partial_fit accumulates observations.
    transform returns a new float64 array without updating statistics.
    statistics_ contains the current per-feature replacement values.
    n_samples_seen_ counts only original nonmissing observations.

    Mean estimation shares the float64 moment-merging helper and its
    numerical limits. No raw sample history is retained.
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self._validate_parameters()

    def _validate_parameters(self):
        if not isinstance(self.strategy, str):
            raise TypeError("strategy must be a string.")
        if self.strategy not in ("mean", "constant"):
            raise ValueError("strategy must be 'mean' or 'constant'.")

        value = np.asarray(self.fill_value)
        if value.ndim != 0 or value.dtype.kind not in "iuf":
            raise TypeError("fill_value must be a real numeric scalar.")

        with np.errstate(over="raise", invalid="raise"):
            fill = float(value)

        if not np.isfinite(fill):
            raise ValueError("fill_value must be finite.")
        return self.strategy, fill

    def fit(self, X, y=None):
        """Reset estimates from nonempty X; y is accepted and ignored."""
        return self._learn(X, reset=True)

    def partial_fit(self, X, y=None):
        """Update estimates using only original nonmissing observations."""
        return self._learn(X, reset=False)

    def _learn(self, X, *, reset):
        strategy, fill = self._validate_parameters()
        fitted = hasattr(self, "n_features_in_")

        if fitted and not reset:
            if (strategy, fill) != self._fitted_config:
                raise ValueError("Parameters changed after fitting; call fit to reset.")

        expected = self.n_features_in_ if fitted and not reset else None
        checked = validate_numeric_X(X, allow_nan=True, n_features=expected)

        previous = None
        if fitted and not reset:
            previous = self._moments

        if strategy == "mean":
            moments = merge_moments(checked, previous)
            count, mean, _ = moments
            statistics = np.where(count > 0, mean, fill)
        else:
            moments = None
            count = np.sum(~np.isnan(checked), axis=0, dtype=np.int64)
            if fitted and not reset:
                limit = np.iinfo(np.int64).max
                if np.any(self.n_samples_seen_ > limit - count):
                    raise OverflowError("Per-feature observation count exceeds int64.")
                count = self.n_samples_seen_ + count

            statistics = np.full(checked.shape[1], fill, dtype=np.float64)

        # Publish only after validation and calculation succeed.
        self.n_features_in_ = checked.shape[1]
        self.n_samples_seen_ = count
        self.statistics_ = statistics
        self._moments = moments
        self._fitted_config = (strategy, fill)
        return self

    def transform(self, X):
        """Replace NaN using fitted estimates without learning from X."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("Imputer must be fitted before transform.")

        if self._validate_parameters() != self._fitted_config:
            raise ValueError("Parameters changed after fitting; call fit to reset.")

        checked = validate_numeric_X(
            X, allow_nan=True, allow_empty=True, n_features=self.n_features_in_,)

        with np.errstate(over="raise", invalid="raise"):
            values = checked.astype(np.float64, copy=True)
            return np.where(np.isnan(values), self.statistics_, values)

    def fit_transform(self, X, y=None):
        """Fit from scratch and transform the same input."""
        return self.fit(X, y).transform(X)

class OneHotEncoder:
    """Incrementally encode categorical features as dense float64 arrays.

    fit resets categories; partial_fit appends newly observed categories.
    Existing category indices within a feature never change. Output blocks
    follow input feature order, so later block offsets can shift on updates.

    categories_ is a list of object arrays. None represents a learned
    missing category; ordinary strings such as 'None' remain distinct.
    handle_unknown='ignore' produces an all-zero block for unknown values.
    handle_unknown='error' rejects unknown values.

    Training requires nonempty input. A fitted transform accepts zero rows.
    String/numeric feature kinds must remain compatible after learning.
    transform neither learns nor modifies caller-owned inputs.
    """

    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown
        self._validate_parameters()

    def _validate_parameters(self):
        if not isinstance(self.handle_unknown, str):
            raise TypeError("handle_unknown must be a string.")
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")

    @staticmethod
    def _lookup(query, categories, missing_index):
        """Map ordinary query categories to stable indices, or -1.

        Search a sorted view while preserving the stored category order.
        Missing categories are excluded from ordinary comparisons.
        """
        result = np.full(query.size, -1, dtype=np.intp)
        ordinary_indices = np.arange(categories.size, dtype=np.intp)
        ordinary_indices = ordinary_indices[ordinary_indices != missing_index]

        if query.size == 0 or ordinary_indices.size == 0:
            return result

        order = np.argsort(categories[ordinary_indices], kind="stable")
        sorted_indices = ordinary_indices[order]
        sorted_categories = categories[sorted_indices]

        positions = np.searchsorted(sorted_categories, query)
        in_bounds = positions < sorted_categories.size
        rows = np.flatnonzero(in_bounds)

        matches = (sorted_categories[positions[rows]] == query[rows])
        rows = rows[matches]
        result[rows] = sorted_indices[positions[rows]]
        return result

    def fit(self, X, y=None):
        """Reset categories from X; y is accepted and ignored."""
        return self._learn(X, reset=True)

    def partial_fit(self, X, y=None):
        """Append new categories without reordering existing categories."""
        return self._learn(X, reset=False)

    def _learn(self, X, *, reset):
        self._validate_parameters()
        fitted = hasattr(self, "n_features_in_")
        reuse = fitted and not reset
        expected = self.n_features_in_ if reuse else None
        columns = validate_categorical_X(X, n_features=expected)

        candidate_categories = []
        candidate_missing = []
        candidate_kinds = []

        for feature, (values, missing, kind) in enumerate(columns):
            if reuse:
                old = self.categories_[feature]
                missing_index = self._missing_indices[feature]
                old_kind = self._feature_kinds[feature]
            else:
                old = np.empty(0, dtype=object)
                missing_index = -1
                old_kind = None

            if (
                old_kind is not None
                and kind is not None
                and kind != old_kind
            ):
                raise TypeError(
                    f"Feature {feature} must retain its "
                    f"{old_kind} category type."
                )

            unique, _ = factorise_column(values, missing)
            known_indices = self._lookup(unique, old, missing_index)
            additions = unique[known_indices < 0]
            updated = np.concatenate((old, additions))

            if missing_index < 0 and np.any(missing):
                missing_index = updated.size
                updated = np.concatenate((
                    updated, np.array([None], dtype=object),
                ))

            candidate_categories.append(updated)
            candidate_missing.append(missing_index)
            candidate_kinds.append(
                old_kind if old_kind is not None else kind
            )

        n_features_out = sum(
            categories.size for categories in candidate_categories
        )

        # Publish only after every feature has been processed successfully.
        self.n_features_in_ = len(columns)
        self.n_features_out_ = n_features_out
        self.categories_ = candidate_categories
        self._missing_indices = candidate_missing
        self._feature_kinds = candidate_kinds
        return self

    def transform(self, X):
        """Return one-hot output without learning or changing categories."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        self._validate_parameters()
        columns = validate_categorical_X(
            X, allow_empty=True, n_features=self.n_features_in_,
        )

        n_samples = columns[0][0].size
        output = np.zeros((n_samples, self.n_features_out_), dtype=np.float64)
        offset = 0

        for feature, (values, missing, kind) in enumerate(columns):
            learned_kind = self._feature_kinds[feature]
            if (
                learned_kind is not None
                and kind is not None
                and kind != learned_kind
            ):
                raise TypeError(
                    f"Feature {feature} must retain its "
                    f"{learned_kind} category type."
                )

            categories = self.categories_[feature]
            missing_index = self._missing_indices[feature]
            unique, local_codes = factorise_column(values, missing)
            category_indices = self._lookup(unique, categories, missing_index)

            codes = np.full(n_samples, -1, dtype=np.intp)
            observed = ~missing
            codes[observed] = category_indices[local_codes[observed]]
            codes[missing] = missing_index

            if (self.handle_unknown == "error" and np.any(codes < 0)):
                raise ValueError(f"Feature {feature} contains unknown categories.")

            rows = np.flatnonzero(codes >= 0)
            output[rows, offset + codes[rows]] = 1.0
            offset += categories.size

        return output

    def fit_transform(self, X, y=None):
        """Fit from scratch and transform the same input."""
        return self.fit(X, y).transform(X)