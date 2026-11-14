"""Cumulative and rolling classification metrics.

Confusion matrix rows represent true labels; columns represent predictions.
"""

from numbers import Integral
import numpy as np
from ._validation import validate_integer, validate_y


def _label_kind(labels):
    """Return the validated label family's name."""
    if labels.dtype.kind in "iu":
        return "integer"
    if labels.dtype.kind == "U":
        return "string"
    return "integer" if isinstance(labels[0], Integral) else "string"


class StreamingConfusionMatrix:
    """Accumulate classification counts across nonempty chunks.

    Parameters
    ----------
    classes : one-dimensional array-like, optional
        Initial unique integer or string labels. Their supplied order is
        preserved. Newly observed labels are appended in sorted order;
        this is an initial class table, not a fixed whitelist.

    Notes
    -----
    Inputs have shape (n_samples,). Labels must be homogeneous integers
    or homogeneous strings, consistently across updates.

    Counting uses NumPy index mapping and bincount, without a Python
    loop over samples. Object-label validation may inspect elements.

    Persistent count storage is O(n_classes ** 2). Raw sample history
    is not retained. Counts use int64; updates exceeding its total
    sample capacity are rejected before state changes.
    """

    def __init__(self, classes=None):
        self._initial_classes = np.empty(0, dtype=object)
        self._initial_kind = None

        if classes is not None:
            checked = validate_y(classes)
            initial = checked.astype(object, copy=True)
            if np.unique(initial).size != initial.size:
                raise ValueError("classes must contain unique labels.")
            self._initial_classes = initial
            self._initial_kind = _label_kind(checked)

        self.reset()

    def reset(self):
        """Clear counts and learned labels, retaining initial classes."""
        classes = self._initial_classes.copy()
        matrix = np.zeros((classes.size, classes.size), dtype=np.int64)

        self._classes = classes
        self._matrix = matrix
        self._kind = self._initial_kind
        self._n_samples = 0
        return self

    @property
    def classes_(self):
        """Return an independent copy of the current class order."""
        return self._classes.copy()

    @property
    def n_samples_seen_(self):
        """Return the total number of evaluated samples."""
        return self._n_samples

    def update(self, y_true, y_pred):
        """Accumulate a nonempty chunk and return self.

        Validation and candidate construction finish before publication.
        A failed update leaves the previous state unchanged.
        """
        truth = validate_y(y_true)
        prediction = validate_y(y_pred)

        if truth.size != prediction.size:
            raise ValueError("y_true and y_pred must contain the same number of samples.")

        kind = _label_kind(truth)
        if _label_kind(prediction) != kind:
            raise TypeError("y_true and y_pred must use the same label type.")
        if self._kind is not None and kind != self._kind:
            raise TypeError("Label type must remain consistent across updates.")

        total = self._n_samples + int(truth.size)
        if total > np.iinfo(np.int64).max:
            raise OverflowError("Cumulative sample count exceeds int64.")

        # Object representation prevents signed/unsigned concatenation
        # from rounding large integer labels through float.
        truth = truth.astype(object, copy=False)
        prediction = prediction.astype(object, copy=False)
        observed = np.unique(np.concatenate((truth, prediction)))

        # Find new labels using a sorted lookup of the existing table.
        # The stored table itself remains in its original order.
        if self._classes.size:
            existing = np.sort(self._classes)
            positions = np.searchsorted(existing, observed)
            known = np.zeros(observed.size, dtype=bool)
            valid = positions < existing.size
            known[valid] = existing[positions[valid]] == observed[valid]
            new_labels = observed[~known]
        else:
            new_labels = observed

        classes = np.concatenate((self._classes, new_labels))
        n_classes = classes.size

        # Map sorted-search positions back to stable matrix indices.
        order = np.argsort(classes, kind="stable")
        sorted_classes = classes[order]
        true_indices = order[np.searchsorted(sorted_classes, truth)]
        pred_indices = order[np.searchsorted(sorted_classes, prediction)]

        flat_indices = true_indices * n_classes + pred_indices
        chunk_counts = np.bincount(flat_indices, minlength=n_classes * n_classes
                                   ).reshape(n_classes, n_classes)

        matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
        old_size = self._classes.size
        matrix[:old_size, :old_size] = self._matrix
        matrix += chunk_counts

        self._classes = classes
        self._matrix = matrix
        self._kind = kind
        self._n_samples = total
        return self

    def result(self):
        """Return independent class/count arrays and the sample count.

        Before any update, counts are zero and n_samples is zero.
        Scalar classification metrics will handle that state separately.
        """
        return {
            "classes": self._classes.copy(),
            "confusion_matrix": self._matrix.copy(),
            "n_samples": self._n_samples,
        }

class StreamingClassificationMetrics(StreamingConfusionMatrix):
    """Classification metrics derived from cumulative confusion counts.

    Parameters
    ----------
    classes : one-dimensional array-like, optional
        Initial class order; new classes are appended during updates.
    zero_division : {0, 1}, default=0
        Per-class value when a metric denominator is zero.
        With no evaluated samples, metrics return NaN regardless of
        this setting.

    Notes
    -----
    Precision, recall and F1 arrays follow the returned class order.
    Macro averages include every class in that table, including absent
    configured classes. Weighted averages use true-label support.

    Counts remain integer-valued; metric calculations use float64.
    No sample history is retained.
    """

    def __init__(self, classes=None, *, zero_division=0):
        if (
            isinstance(zero_division, (bool, np.bool_))
            or not isinstance(zero_division, (Integral, float, np.floating))
        ):
            raise TypeError("zero_division must be numeric 0 or 1.")
        if zero_division not in (0, 1):
            raise ValueError("zero_division must be 0 or 1.")

        self._zero_division = float(zero_division)
        super().__init__(classes=classes)

    @property
    def zero_division(self):
        """Return the configured zero-denominator value."""
        return self._zero_division

    def result(self):
        """Return counts, per-class metrics and macro/weighted averages.

        support contains true-label counts; predicted_support contains
        predicted-label counts. All returned arrays are independent of
        the accumulator's state. This method does not learn.
        """
        result = super().result()
        matrix = result["confusion_matrix"]
        n_samples = result["n_samples"]

        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)

        # Convert before multiplication/addition to avoid int64 overflow
        # in expressions such as 2 * TP or support + predicted_support.
        true_positive = np.diag(matrix).astype(np.float64)
        true_count = support.astype(np.float64)
        predicted_count = predicted_support.astype(np.float64)

        if n_samples == 0:
            precision = np.full(matrix.shape[0], np.nan)
            recall = np.full(matrix.shape[0], np.nan)
            f1 = np.full(matrix.shape[0], np.nan)
            accuracy = float("nan")
        else:
            precision = np.full(matrix.shape[0], self._zero_division)
            recall = np.full(matrix.shape[0], self._zero_division)
            f1 = np.full(matrix.shape[0], self._zero_division)

            np.divide(
                true_positive,
                predicted_count,
                out=precision,
                where=predicted_count != 0,
            )
            np.divide(
                true_positive,
                true_count,
                out=recall,
                where=true_count != 0,
            )
            f1_denominator = true_count + predicted_count
            np.divide(
                2.0 * true_positive,
                f1_denominator,
                out=f1,
                where=f1_denominator != 0,
            )
            accuracy = float(true_positive.sum() / n_samples)

        result.update({
            "support": support,
            "predicted_support": predicted_support,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        # This loop covers three metric names, not individual samples.
        for name, values in (
            ("precision", precision),
            ("recall", recall),
            ("f1", f1),
        ):
            if n_samples == 0:
                macro = weighted = float("nan")
            else:
                macro = float(values.mean())
                weights = true_count / n_samples
                weighted = float(np.dot(values, weights))
            result[f"{name}_macro"] = macro
            result[f"{name}_weighted"] = weighted

        return result

def binary_roc_auc(y_true, y_score, *, positive_label):
    """Compute binary ROC AUC using sorted score groups.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Homogeneous integer or string labels.
    y_score : array-like of shape (n_samples,)
        Finite real scores. Larger scores indicate the positive class.
        Integer NumPy arrays are sorted without conversion to float.
        Array-like inputs follow NumPy dtype inference.
    positive_label : integer or string
        Explicit positive class. At most one other label is allowed.

    Returns
    -------
    float
        AUC, or NaN when either class has no samples.

    Notes
    -----
    Equal scores receive half credit. Sorting costs O(n log n);
    auxiliary arrays require O(n) memory. No pairwise score matrix
    is constructed. Inputs are not modified.
    """
    positive = validate_y([positive_label])
    truth = validate_y(y_true, allow_empty=True)
    scores = np.asarray(y_score)

    if scores.ndim != 1:
        raise ValueError("y_score must be a one-dimensional array.")
    if scores.size != truth.size:
        raise ValueError("y_true and y_score must contain the same number of samples.")
    if scores.dtype.kind not in "iuf":
        raise TypeError("y_score must contain real numeric scores.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite scores.")

    if truth.size == 0:
        return float("nan")

    if _label_kind(truth) != _label_kind(positive):
        raise TypeError("positive_label must use the same label type as y_true.")

    # Preserve exact integer labels when signed and unsigned values mix.
    labels = truth.astype(object, copy=False)
    is_positive = labels == positive[0]
    negative_labels = np.unique(labels[~is_positive])
    if negative_labels.size > 1:
        raise ValueError("Binary AUC supports at most one label besides positive_label.")

    n_positive = int(np.count_nonzero(is_positive))
    n_negative = int(truth.size) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = is_positive[order].astype(np.int64)

    # Group boundaries use equality, avoiding score subtraction overflow.
    starts = np.concatenate((
        np.array([0], dtype=np.intp),
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
    ))
    ends = np.concatenate((starts[1:], np.array([scores.size], dtype=np.intp),))

    positive_counts = np.add.reduceat(sorted_positive, starts)
    negative_counts = (ends - starts) - positive_counts
    negatives_before = np.cumsum(negative_counts) - negative_counts

    # Each positive beats all lower-scoring negatives and receives
    # half credit against negatives in its own score group.
    wins = np.dot(
        positive_counts.astype(np.float64),
        negatives_before.astype(np.float64)
        + 0.5 * negative_counts.astype(np.float64),
    )
    return float(wins / (n_positive * n_negative))

class StreamingROCAUC:
    """Exact cumulative binary AUC over retained evaluation history.

    Parameters
    ----------
    positive_label : integer or string
        Required positive class. Larger scores indicate this class.

    Notes
    -----
    update accepts nonempty chunks. Only one negative label is allowed
    across all updates since reset. The positive label is fixed.

    Scores are stored as float64. Integer scores must be exactly
    representable in float64; otherwise the update is rejected.
    Higher-precision floating scores follow float64 rounding.

    Label and score arrays are copied. Persistent memory grows with
    evaluated samples and chunk count. Updates copy chunk-reference
    lists, not previous sample arrays. result concatenates and sorts
    retained history, requiring O(n) auxiliary memory and O(n log n)
    time. This is not a bounded-memory estimator.
    """

    def __init__(self, *, positive_label):
        positive = validate_y([positive_label])
        self._positive_label = positive[0]
        self._kind = _label_kind(positive)
        self.reset()

    @property
    def positive_label(self):
        """Return the configured positive label."""
        return self._positive_label

    @property
    def n_samples_seen_(self):
        """Return the number of retained evaluated samples."""
        return self._n_samples

    def reset(self):
        """Clear evaluation history while retaining the positive class."""
        self._label_chunks = []
        self._score_chunks = []
        self._negative_label = None
        self._n_samples = 0
        self._n_positive = 0
        return self

    def update(self, y_true, y_score):
        """Validate and retain a nonempty chunk, then return self.

        All candidate arrays and lists are prepared before publication.
        Validation or preparation failure leaves existing state intact.
        """
        truth = validate_y(y_true)
        if _label_kind(truth) != self._kind:
            raise TypeError("y_true must use the same label type as positive_label.")

        scores = np.asarray(y_score)
        if scores.ndim != 1:
            raise ValueError("y_score must be a one-dimensional array.")
        if scores.size != truth.size:
            raise ValueError(
                "y_true and y_score must contain the same number of samples."
            )
        if scores.dtype.kind not in "iuf":
            raise TypeError("y_score must contain real numeric scores.")
        if not np.isfinite(scores).all():
            raise ValueError("y_score must contain only finite scores.")

        labels = truth.astype(object, copy=True)
        positive_mask = labels == self._positive_label
        negatives = np.unique(labels[~positive_mask])
        if negatives.size > 1:
            raise ValueError("Binary AUC supports at most one negative label.")

        negative_label = self._negative_label
        if negatives.size:
            candidate_negative = negatives[0]
            if (
                negative_label is not None and candidate_negative != negative_label
            ):
                raise ValueError("The negative label must remain consistent until reset.")
            negative_label = candidate_negative

        with np.errstate(over="ignore", invalid="ignore"):
            stored_scores = scores.astype(np.float64, copy=True)
        if not np.isfinite(stored_scores).all():
            raise ValueError("Scores must remain finite in float64.")

        if scores.dtype.kind in "iu":
            # Object comparisons use exact Python integer/float equality,
            # avoiding NumPy's implicit integer-to-float comparison cast.
            exact = (
                stored_scores.astype(object) == scores.astype(object)
            )
            if not np.all(exact):
                raise ValueError("Integer scores must be exactly representable in float64.")

        label_chunks = self._label_chunks + [labels]
        score_chunks = self._score_chunks + [stored_scores]
        n_samples = self._n_samples + int(labels.size)
        n_positive = (self._n_positive + int(np.count_nonzero(positive_mask)))

        self._label_chunks = label_chunks
        self._score_chunks = score_chunks
        self._negative_label = negative_label
        self._n_samples = n_samples
        self._n_positive = n_positive
        return self

    def result(self):
        """Return cumulative AUC and evaluated class counts.

        AUC is NaN before evaluation or while either class is absent.
        Querying does not modify retained history or counts.
        """
        n_negative = self._n_samples - self._n_positive
        if self._n_positive == 0 or n_negative == 0:
            auc = float("nan")
        else:
            labels = np.concatenate(self._label_chunks)
            scores = np.concatenate(self._score_chunks)
            auc = binary_roc_auc(labels, scores, positive_label=self._positive_label)

        return {
            "auc": auc,
            "n_samples": self._n_samples,
            "n_positive": self._n_positive,
            "n_negative": n_negative,
        }

class RollingClassificationMetrics:
    """Classification metrics over the latest window_size samples.

    Every incoming chunk is validated in full before window truncation.
    Observed class order is retained until reset, even when a class
    leaves the window. Macro averages include those absent classes.

    Persistent sample storage is O(window_size). Class tables and dense
    confusion counts additionally require O(n_classes ** 2) storage.
    Updates recount the retained window; they are not constant-time.
    Temporary arrays also depend on the incoming chunk size.
    """

    def __init__(self, window_size, *, classes=None, zero_division=0):
        self._window_size = validate_integer(window_size, "window_size", minimum=1)
        initial = StreamingClassificationMetrics(
            classes=classes, zero_division=zero_division
        )
        self._initial_classes = initial.classes_
        self._zero_division = initial.zero_division
        self.reset()

    @property
    def window_size(self):
        """Return the configured maximum retained sample count."""
        return self._window_size

    @property
    def zero_division(self):
        """Return the configured zero-denominator value."""
        return self._zero_division

    @property
    def classes_(self):
        """Return an independent copy of the stable class table."""
        return self._metrics.classes_

    @property
    def n_samples_seen_(self):
        """Return all successfully received samples since reset."""
        return self._n_samples_seen

    def reset(self):
        """Clear the window, learned classes and received-sample count."""
        classes = (
            self._initial_classes
            if self._initial_classes.size
            else None
        )
        metrics = StreamingClassificationMetrics(
            classes=classes, zero_division=self._zero_division
        )
        truth = np.empty(0, dtype=object)
        prediction = np.empty(0, dtype=object)

        self._metrics = metrics
        self._truth = truth
        self._prediction = prediction
        self._n_samples_seen = 0
        return self

    def update(self, y_true, y_pred):
        """Retain the newest samples and publish a complete new state."""
        truth = validate_y(y_true)
        prediction = validate_y(y_pred)

        # Validate the entire incoming chunk, including any prefix that
        # will immediately leave the window. Preserve all observed
        # classes and their stable order.
        current_classes = self._metrics.classes_
        schema = StreamingConfusionMatrix(
            classes=current_classes if current_classes.size else None
        )
        schema.update(truth, prediction)

        incoming_size = int(truth.size)
        if incoming_size >= self._window_size:
            retained_truth = truth[-self._window_size:].astype(
                object, copy=True
            )
            retained_prediction = prediction[-self._window_size:].astype(
                object, copy=True
            )
        else:
            keep_old = self._window_size - incoming_size
            retained_truth = np.concatenate((
                self._truth[-keep_old:],
                truth.astype(object, copy=False),
            ))
            retained_prediction = np.concatenate((
                self._prediction[-keep_old:],
                prediction.astype(object, copy=False),
            ))

        candidate = StreamingClassificationMetrics(
            classes=schema.classes_,
            zero_division=self._zero_division,
        )
        candidate.update(retained_truth, retained_prediction)
        n_samples_seen = self._n_samples_seen + incoming_size

        self._metrics = candidate
        self._truth = retained_truth
        self._prediction = retained_prediction
        self._n_samples_seen = n_samples_seen
        return self

    def result(self):
        """Return window metrics, capacity and total received count.

        n_samples is the current window population.
        n_samples_seen is the total received since reset.
        Returned arrays are independent of retained state.
        """
        result = self._metrics.result()
        result["window_size"] = self._window_size
        result["n_samples_seen"] = self._n_samples_seen
        return result

class RollingROCAUC(StreamingROCAUC):
    """Binary ROC AUC over the latest window_size evaluated samples.

    The positive class and the first observed negative class remain
    fixed until reset, even when either class leaves the window.

    All incoming samples are validated before truncation. Score storage
    follows StreamingROCAUC's float64 and integer-exactness rules.

    Retained sample storage is O(window_size). Temporary update storage
    also depends on incoming chunk size. Queries sort the retained
    window, taking O(w log w) time and O(w) auxiliary memory.
    """

    def __init__(self, window_size, *, positive_label):
        self._window_size = validate_integer(
            window_size, "window_size", minimum=1
        )
        super().__init__(positive_label=positive_label)

    @property
    def window_size(self):
        """Return the maximum retained sample count."""
        return self._window_size

    @property
    def n_samples_seen_(self):
        """Return all successfully received samples since reset."""
        return self._n_samples_seen

    def reset(self):
        """Clear the window and negative class, retaining configuration."""
        super().reset()
        self._n_samples_seen = 0
        return self

    def update(self, y_true, y_score):
        """Validate a full chunk and retain only the newest window."""
        # A temporary accumulator validates and owns the incoming chunk.
        # It inherits the stream's negative-class constraint, but none
        # of its previous sample history.
        incoming = StreamingROCAUC(positive_label=self._positive_label)
        incoming._negative_label = self._negative_label
        incoming.update(y_true, y_score)

        labels = incoming._label_chunks[0]
        scores = incoming._score_chunks[0]
        incoming_size = incoming.n_samples_seen_

        if incoming_size >= self._window_size:
            # Copies prevent a small retained tail from keeping the
            # entire oversized input allocation alive.
            retained_labels = labels[-self._window_size:].copy()
            retained_scores = scores[-self._window_size:].copy()
        else:
            keep_old = self._window_size - incoming_size
            old_labels = (
                self._label_chunks[0]
                if self._label_chunks
                else np.empty(0, dtype=object)
            )
            old_scores = (
                self._score_chunks[0]
                if self._score_chunks
                else np.empty(0, dtype=np.float64)
            )
            retained_labels = np.concatenate((
                old_labels[-keep_old:], labels,
            ))
            retained_scores = np.concatenate((
                old_scores[-keep_old:], scores,
            ))

        label_chunks = [retained_labels]
        score_chunks = [retained_scores]
        n_samples = int(retained_labels.size)
        n_positive = int(np.count_nonzero(
            retained_labels == self._positive_label
        ))
        n_samples_seen = self._n_samples_seen + incoming_size

        # Publish only after all candidate arrays and counts exist.
        self._label_chunks = label_chunks
        self._score_chunks = score_chunks
        self._negative_label = incoming._negative_label
        self._n_samples = n_samples
        self._n_positive = n_positive
        self._n_samples_seen = n_samples_seen
        return self

    def result(self):
        """Return window AUC, window counts and total received count.

        n_samples, n_positive and n_negative describe only the window.
        n_samples_seen counts all successful input since reset.
        """
        result = super().result()
        result["window_size"] = self._window_size
        result["n_samples_seen"] = self._n_samples_seen
        return result