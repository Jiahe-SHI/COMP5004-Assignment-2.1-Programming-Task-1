"""Numeric decision tree construction and vectorised split search."""

from dataclasses import dataclass
from math import isqrt

import numpy as np

from ._validation import (
    validate_integer,
    validate_numeric_X,
    validate_tree_parameters,
    validate_X_y,
    validate_y,
)


def _encode_labels(y):
    """Return sorted classes and integer codes for nonempty labels.

    Parameters
    ----------
    y : array-like, shape (n_samples,)
        Homogeneous integer or Unicode string labels.

    Returns
    -------
    classes : ndarray, shape (n_classes,)
        Sorted distinct labels, without conversion through float.
    encoded : ndarray, shape (n_samples,)
        Integer indices satisfying classes[encoded] == y.

    Notes
    -----
    This function creates a fresh class table for batch training.
    It does not implement cumulative class-table extension.
    """
    labels = validate_y(y)
    classes, encoded = np.unique(labels, return_inverse=True)
    return classes, encoded.astype(np.intp, copy=False)


def _impurity_from_counts(counts, criterion="gini"):
    """Compute impurity from nonnegative class counts.

    Parameters
    ----------
    counts : array-like
        Shape (n_classes,) for one node, or (n_nodes, n_classes)
        for multiple nodes. The last axis represents classes.
        Integer and floating counts are accepted.
    criterion : {"gini", "entropy"}
        Entropy uses logarithm base 2.

    Returns
    -------
    float or ndarray, shape (n_nodes,)
        A scalar for one-dimensional input, otherwise one value per node.

    Notes
    -----
    All-zero rows return zero as a numerical convention. The split
    search must still exclude candidates with empty child nodes.

    Counts are scaled before summation to avoid overflow when normalising
    large finite values. Inputs are never modified.
    """
    if not isinstance(criterion, str):
        raise TypeError("criterion must be a string.")
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'.")

    values = np.asarray(counts)

    if values.ndim not in (1, 2):
        raise ValueError("counts must be one- or two-dimensional.")
    if values.shape[-1] == 0:
        raise ValueError("counts must contain at least one class.")
    if values.dtype.kind not in "iuf":
        raise TypeError("counts must contain real numeric values.")
    if not np.isfinite(values).all():
        raise ValueError("counts must contain only finite values.")
    if (values < 0).any():
        raise ValueError("counts must be nonnegative.")

    # Use a private floating array, preserving caller-owned input.
    work = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(work).all():
        raise ValueError("counts must be representable as finite float64.")

    largest = np.max(work, axis=-1, keepdims=True)
    scaled = np.zeros_like(work)
    np.divide(work, largest, out=scaled, where=largest > 0)

    totals = np.sum(scaled, axis=-1, keepdims=True)
    probabilities = np.zeros_like(scaled)
    np.divide(scaled, totals, out=probabilities, where=totals > 0)

    if criterion == "gini":
        impurity = 1.0 - np.sum(probabilities**2, axis=-1)
    else:
        log_probabilities = np.zeros_like(probabilities)
        np.log2(probabilities, out=log_probabilities, where=probabilities > 0,)
        impurity = -np.sum(probabilities * log_probabilities, axis=-1)

    impurity = np.where(totals[..., 0] > 0, impurity, 0.0)

    if values.ndim == 1:
        return float(impurity)
    return impurity

# Absolute tolerance for gain comparisons and numerical zero.
_GAIN_TOL = 1e-12


def _split_threshold(lower, upper):
    """Return a threshold t with lower <= t < upper.

    Inputs are adjacent distinct values from one validated numeric column.
    Integer arithmetic avoids overflow and preserves large integer values.
    Floating midpoints fall back to the lower endpoint when rounded up.
    """
    dtype = np.asarray(lower).dtype
    if dtype.kind in "iu":
        return (int(lower) + int(upper)) // 2

    with np.errstate(under="ignore"):
        midpoint = dtype.type(lower / 2 + upper / 2)
    if not lower <= midpoint < upper:
        midpoint = lower
    return midpoint.item()


def _best_split(
    X, encoded_y, n_classes, *, criterion="gini", feature_indices=None):
    """Return the best positive-gain split, or None.

    X has shape (n_samples, n_features). encoded_y is a one-dimensional
    integer array with values in [0, n_classes).

    All candidate positions within each feature are evaluated in bulk.
    Features are visited in ascending index order. Gains within 1e-12
    of the global maximum are tied; choose the first feature and then
    the first split position. Gains <= 1e-12 are numerical zero.

    The returned dict contains feature, threshold, gain, n_left, n_right.
    The partition rule is X[:, feature] <= threshold.
    No input is modified. Feature subsampling is the caller's responsibility.
    """
    X = validate_numeric_X(X)
    n_classes = validate_integer(n_classes, "n_classes", minimum=1)
    codes = np.asarray(encoded_y)

    if codes.ndim != 1 or codes.size != X.shape[0]:
        raise ValueError("encoded_y must be one-dimensional and match X.")
    if codes.dtype.kind not in "iu":
        raise TypeError("encoded_y must contain integer class indices.")
    if (codes < 0).any() or (codes >= n_classes).any():
        raise ValueError("encoded_y contains an out-of-range class index.")
    codes = codes.astype(np.intp, copy=False)

    if feature_indices is None:
        features = np.arange(X.shape[1])
    else:
        features = np.asarray(feature_indices)
        if features.ndim != 1:
            raise ValueError("feature_indices must be one-dimensional.")
        if features.dtype.kind not in "iu":
            raise TypeError("feature_indices must contain integers.")
        if (features < 0).any() or (features >= X.shape[1]).any():
            raise ValueError("feature_indices contains an invalid index.")
        if np.unique(features).size != features.size:
            raise ValueError("feature_indices must not contain duplicates.")
        features = np.sort(features)

    n_samples = X.shape[0]
    total_counts = np.bincount(codes, minlength=n_classes)
    parent_impurity = _impurity_from_counts(total_counts, criterion)
    if n_samples < 2 or np.count_nonzero(total_counts) < 2:
        return None

    records = []
    maximum_gain = 0.0

    # Feature-level iteration; no Python loop over samples or thresholds.
    for feature in features:
        feature = int(feature)
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        positions = np.flatnonzero(values[:-1] < values[1:])
        if positions.size == 0:
            continue

        cumulative = np.zeros((n_samples, n_classes), dtype=np.int64)
        cumulative[np.arange(n_samples), codes[order]] = 1
        np.cumsum(cumulative, axis=0, out=cumulative)

        left_counts = cumulative[positions]
        right_counts = total_counts - left_counts
        left_sizes = positions + 1
        right_sizes = n_samples - left_sizes

        gains = (
            parent_impurity - (left_sizes / n_samples)
            * _impurity_from_counts(left_counts, criterion)
            - (right_sizes / n_samples)
            * _impurity_from_counts(right_counts, criterion)
        )
        maximum_gain = max(maximum_gain, float(np.max(gains)))
        records.append((feature, values, positions, gains))

    if maximum_gain <= _GAIN_TOL:
        return None

    # Resolve ties against the global maximum, not a moving local best.
    for feature, values, positions, gains in records:
        eligible = np.flatnonzero(
            (gains > _GAIN_TOL) & (gains >= maximum_gain - _GAIN_TOL)
        )
        if eligible.size:
            candidate = int(eligible[0])
            position = int(positions[candidate])
            return {
                "feature": feature,
                "threshold": _split_threshold(
                    values[position], values[position + 1]
                ),
                "gain": float(gains[candidate]),
                "n_left": position + 1,
                "n_right": n_samples - position - 1,
            }

    return None

def _threshold_mask(values, threshold):
    """Compare finite numeric values with a stored threshold accurately.

    Avoid NumPy scalar promotion rounding a threshold across a data value
    when prediction and training use different numeric dtypes.
    """
    if values.dtype.kind in "iu":
        if isinstance(threshold, (int, np.integer)):
            cutoff = int(threshold)
        else:
            cutoff = int(np.floor(threshold))
        limits = np.iinfo(values.dtype)
        if cutoff < limits.min:
            return np.zeros(values.shape, dtype=bool)
        if cutoff >= limits.max:
            return np.ones(values.shape, dtype=bool)
        return values <= values.dtype.type(cutoff)

    # Round to the prediction dtype, then move down if rounding went up.
    with np.errstate(over="ignore", under="ignore"):
        rounded = values.dtype.type(threshold)
        if np.isposinf(rounded):
            return np.ones(values.shape, dtype=bool)
        if np.isneginf(rounded):
            return np.zeros(values.shape, dtype=bool)
        if rounded.item() > threshold:
            rounded = np.nextafter(rounded, values.dtype.type(-np.inf))
    return values <= rounded

@dataclass(eq=False)
class _TreeNode:
    """Internal node; class counts use the fitted classes_ column order."""

    counts: np.ndarray
    prediction: int
    n_samples: int
    depth: int
    impurity: float
    feature: int | None = None
    threshold: int | float | None = None
    gain: float = 0.0
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTreeClassifier:
    """A batch decision tree classifier using numeric threshold splits.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth, with root depth zero. None imposes no explicit limit.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a node split.
    max_features : None, int, "sqrt", or "log2", default=None
        Number of randomly selected candidate features at each split.
        None uses all features. Named options round down, with minimum one.
    buffer_size : int, default=500
        Reserved threshold for buffered incremental training.
        Batch fit always trains on all supplied rows immediately.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity criterion.
    random_state : int or None, default=None
        Seed for a local generator, recreated on each fit.

    Notes
    -----
    Features must be finite real numeric values. Labels must be homogeneous
    integers or strings. Each successful fit replaces the previous model
    and sorts its own class table. Failed fitting preserves learned state.

    Selected constant features are not replaced with unselected features.
    If the selected subset has no positive-gain split, the node is a leaf.
    Recursive construction is subject to Python's recursion limit.
    """

    def __init__(self, *, max_depth=None, min_samples_split=2, max_features=None,
        buffer_size=500, criterion="gini", random_state=None,):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.criterion = criterion
        self.random_state = random_state
        self._validate_parameters()

    @property
    def is_fitted_(self):
        """Whether a complete tree has been fitted successfully."""
        return hasattr(self, "tree_")

    def _validate_parameters(self, n_features=None):
        validate_tree_parameters(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            buffer_size=self.buffer_size,
            criterion=self.criterion,
            random_state=self.random_state,
            n_features=n_features,
        )

    def _feature_count(self, n_features):
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, isqrt(n_features))
        if self.max_features == "log2":
            return max(1, n_features.bit_length() - 1)
        return int(self.max_features)

    def fit(self, X, y):
        """Build a fresh tree from all rows and return self.

        X has shape (n_samples, n_features); y has shape (n_samples,).
        Validation and complete candidate construction precede publication.
        Training arrays are not retained by the fitted model.
        """
        X_checked, y_checked = validate_X_y(X, y)
        n_features = X_checked.shape[1]
        self._validate_parameters(n_features)

        classes, encoded = _encode_labels(y_checked)
        n_classes = int(classes.size)
        feature_count = self._feature_count(n_features)
        seed = None if self.random_state is None else int(self.random_state)
        rng = np.random.default_rng(seed)

        candidate = self._build_tree(
            X_checked, encoded, n_classes, 0, rng, feature_count
        )
        classes.setflags(write=False)

        # Publish only after the whole candidate tree has been built.
        self.__dict__.update(
            classes_=classes,
            n_classes_=n_classes,
            n_features_in_=n_features,
            max_features_=feature_count,
            n_samples_fit_=X_checked.shape[0],
            tree_=candidate,
        )
        return self

    def _build_tree(self, X, codes, n_classes, depth, rng, feature_count):
        counts = np.bincount(codes, minlength=n_classes)
        counts.setflags(write=False)
        node = _TreeNode(
            counts=counts,
            prediction=int(np.argmax(counts)),
            n_samples=int(codes.size),
            depth=depth,
            impurity=_impurity_from_counts(counts, self.criterion),
        )

        if (np.count_nonzero(counts) == 1
            or codes.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        if feature_count == X.shape[1]:
            features = np.arange(X.shape[1])
        else:
            features = np.sort(
                rng.choice(X.shape[1], size=feature_count, replace=False)
            )

        split = _best_split(
            X, codes, n_classes,
            criterion=self.criterion,
            feature_indices=features,
        )
        if split is None:
            return node

        mask = _threshold_mask(X[:, split["feature"]], split["threshold"])
        actual_left = int(np.count_nonzero(mask))
        if actual_left != split["n_left"] or not 0 < actual_left < codes.size:
            raise RuntimeError("Split threshold does not match its partition.")

        node.left = self._build_tree(
            X[mask], codes[mask], n_classes, depth + 1, rng, feature_count
        )
        node.right = self._build_tree(
            X[~mask], codes[~mask], n_classes, depth + 1, rng, feature_count
        )
        node.feature = split["feature"]
        node.threshold = split["threshold"]
        node.gain = split["gain"]
        return node

    def _validate_prediction_X(self, X):
        if not self.is_fitted_:
            raise RuntimeError("DecisionTreeClassifier is not fitted. Call fit before prediction.")

        return validate_numeric_X(X, allow_empty=True, n_features=self.n_features_in_)

    def _leaf_batches(self, X):
        """Yield each reached leaf and its original prediction row indices."""
        if X.shape[0] == 0:
            return

        stack = [(self.tree_, np.arange(X.shape[0]))]
        while stack:
            node, indices = stack.pop()
            if node.is_leaf:
                yield node, indices
                continue

            mask = _threshold_mask(X[indices, node.feature], node.threshold)
            left_indices = indices[mask]
            right_indices = indices[~mask]
            if right_indices.size:
                stack.append((node.right, right_indices))
            if left_indices.size:
                stack.append((node.left, left_indices))

    def predict(self, X):
        """Return original class labels with shape (n_samples,).

        X must be a finite numeric matrix with the fitted feature count.
        Empty input with shape (0, n_features_in_) is supported.
        Prediction does not update learned state. Vote ties follow classes_.
        """
        X_checked = self._validate_prediction_X(X)
        result = np.empty(X_checked.shape[0], dtype=self.classes_.dtype)
        for node, indices in self._leaf_batches(X_checked):
            result[indices] = self.classes_[node.prediction]
        return result

    def predict_proba(self, X):
        """Return leaf class frequencies with shape (n_samples, n_classes_).

        Column j corresponds to classes_[j]. Probabilities are empirical
        training frequencies in the reached leaf, without smoothing.
        Empty input returns an array of shape (0, n_classes_).
        """
        X_checked = self._validate_prediction_X(X)
        result = np.empty((X_checked.shape[0], self.n_classes_), dtype=np.float64)

        for node, indices in self._leaf_batches(X_checked):
            result[indices] = node.counts / node.n_samples
        return result