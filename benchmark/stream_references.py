"""Independent loop references for streaming benchmarks."""

from numbers import Integral
import numpy as np
from numcompute_stream._validation import validate_y


def loop_confusion_matrix(y_true, y_pred):
    """Build batch confusion counts using a Python sample loop.

    Inputs are nonempty one-dimensional homogeneous integer or string
    labels. Both inputs must have equal lengths and matching label types.

    Returns a dictionary containing sorted classes, int64 counts and
    the sample count. Rows are true labels; columns are predictions.

    This is a batch benchmark reference, not a stateful streaming API.
    Input validation is shared with the package; class discovery,
    mapping and counting are implemented independently.
    """
    truth = validate_y(y_true)
    prediction = validate_y(y_pred)

    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must contain the same number of samples.")

    true_integer = isinstance(truth[0], Integral)
    pred_integer = isinstance(prediction[0], Integral)
    if true_integer != pred_integer:
        raise TypeError("y_true and y_pred must use the same label type.")

    # Convert directly to Python objects, preserving large integers.
    true_values = truth.tolist()
    pred_values = prediction.tolist()
    labels = sorted(set(true_values) | set(pred_values))
    positions = {label: index for index, label in enumerate(labels)}

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(true_values, pred_values):
        matrix[positions[actual], positions[predicted]] += 1

    return {
        "classes": np.array(labels, dtype=object),
        "confusion_matrix": matrix,
        "n_samples": int(truth.size),
    }

def loop_best_split(X, encoded_y, n_classes, *, criterion="gini"):
    """Find a split with scalar count updates and impurity calculations.

    Benchmark scope: all features, finite numeric X, encoded integer
    labels. Sorting uses NumPy, as in the production implementation.
    Candidate evaluation uses Python loops.

    Return feature, boundary, gain, n_left and n_right, or None.
    boundary is the largest observed value on the left. It describes
    the training partition, not the production midpoint threshold.
    """
    import math

    from numcompute_stream._validation import (validate_integer, validate_numeric_X,)

    X = validate_numeric_X(X)
    n_classes = validate_integer(n_classes, "n_classes", minimum=1)
    codes = np.asarray(encoded_y)

    if codes.ndim != 1 or codes.size != X.shape[0]:
        raise ValueError("encoded_y must be one-dimensional and match X.")
    if codes.dtype.kind not in "iu":
        raise TypeError("encoded_y must contain integer class indices.")
    if (codes < 0).any() or (codes >= n_classes).any():
        raise ValueError("encoded_y contains an out-of-range class index.")
    if not isinstance(criterion, str):
        raise TypeError("criterion must be a string.")
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be 'gini' or 'entropy'.")

    def impurity(counts, size):
        if criterion == "gini":
            return 1.0 - sum((count / size) ** 2 for count in counts)
        return -sum(
            (count / size) * math.log2(count / size)
            for count in counts if count
        )

    n_samples = X.shape[0]
    total = [0] * n_classes
    for code in codes:
        total[int(code)] += 1

    if n_samples < 2 or sum(count > 0 for count in total) < 2:
        return None

    parent = impurity(total, n_samples)
    candidates = []

    for feature in range(X.shape[1]):
        order = np.argsort(X[:, feature], kind="stable")
        values = X[order, feature]
        left = [0] * n_classes
        right = total.copy()

        for position in range(n_samples - 1):
            code = int(codes[order[position]])
            left[code] += 1
            right[code] -= 1

            if values[position] == values[position + 1]:
                continue

            n_left = position + 1
            n_right = n_samples - n_left
            gain = (
                parent
                - (n_left / n_samples) * impurity(left, n_left)
                - (n_right / n_samples) * impurity(right, n_right)
            )
            candidates.append({
                "feature": feature,
                "boundary": values[position].item(),
                "gain": gain,
                "n_left": n_left,
                "n_right": n_right,
            })

    maximum = max((candidate["gain"] for candidate in candidates), default=0.0)
    tolerance = 1e-12
    if maximum <= tolerance:
        return None

    # Resolve ties against the global maximum, not a moving best.
    for candidate in candidates:
        if (
            candidate["gain"] > tolerance
            and candidate["gain"] >= maximum - tolerance
        ):
            return candidate

    return None