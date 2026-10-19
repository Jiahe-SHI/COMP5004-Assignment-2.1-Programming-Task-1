"""assess the metrics for classification and regression using numpy."""

import numpy as np
from numbers import Real

def _validate_label_values(array, name):
    """Validate label values and return their common kind."""
    label_kind = None

    for value in array:
        if value is None:
            raise ValueError(
                f"{name} must not contain missing labels."
            )

        if isinstance(value, (str, np.str_)):
            current_kind = "string"

        elif isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"{name} labels must be strings or real numbers."
            )

        elif isinstance(
            value,
            (Real, np.integer, np.floating),
        ):
            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must not contain NaN or infinity."
                )

            current_kind = "numeric"

        else:
            raise TypeError(
                f"{name} labels must be strings or real numbers."
            )

        if label_kind is None:
            label_kind = current_kind

        elif current_kind != label_kind:
            raise TypeError(
                f"{name} must contain one homogeneous type."
            )

    return label_kind


def _validate_classification_inputs(y_true, y_pred):
    """Validate and return one-dimensional classification labels."""
    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have equal lengths.")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    true_kind = _validate_label_values(true, "y_true")
    pred_kind = _validate_label_values(pred, "y_pred")

    if true_kind != pred_kind:
        raise TypeError(
            "y_true and y_pred labels must have "
            "the same homogeneous type."
        )

    return true, pred

def _validate_metric_options(average, zero_division):
    """Validate shared classification metric options."""
    if not isinstance(average, str):
        raise TypeError("average must be a string.")

    valid_averages = (
        "binary",
        "macro",
        "micro",
        "weighted",
    )

    if average not in valid_averages:
        raise ValueError(
            "average must be 'binary', 'macro', "
            "'micro' or 'weighted'."
        )

    if isinstance(zero_division, (bool, np.bool_)):
        raise TypeError("zero_division must be 0 or 1.")

    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1.")
    
def accuracy(y_true, y_pred):
    """Return the fraction of matching classification labels."""
    true, pred = _validate_classification_inputs(
        y_true,
        y_pred,
    )

    return float(np.mean(true == pred))

def confusion_matrix(y_true, y_pred, labels=None):
    """Return counts arranged by true and predicted labels."""
    true, pred = _validate_classification_inputs(
        y_true,
        y_pred,
    )

    if labels is None:
        ordered_labels = []

        for label in np.concatenate((true, pred)):
            if label not in ordered_labels:
                ordered_labels.append(label)

        label_array = np.asarray(ordered_labels)
    else:
        label_array = np.asarray(labels, dtype=object)

        if label_array.ndim != 1:
            raise ValueError("labels must be 1D.")

        if label_array.size == 0:
            raise ValueError("labels must not be empty.")

        label_kind = _validate_label_values(
            label_array,
            "labels",
        )
        observed_kind = _validate_label_values(
            true,
            "y_true",
        )

        if label_kind != observed_kind:
            raise TypeError(
                "labels must have the same type as "
                "the observed labels."
            )

        label_list = label_array.tolist()

        if len(set(label_list)) != len(label_list):
            raise ValueError(
                "labels must contain unique values."
            )
        
    label_to_index = {
        label: index
        for index, label in enumerate(label_array.tolist())
    }

    observed_labels = np.concatenate((true, pred)).tolist()

    if any(
        label not in label_to_index
        for label in observed_labels
    ):
        raise ValueError(
            "labels must include all observed labels."
        )

    matrix = np.zeros(
        (label_array.size, label_array.size),
        dtype=np.int64,
    )

    for actual, predicted in zip(true.tolist(), pred.tolist()):
        matrix[
            label_to_index[actual],
            label_to_index[predicted],
        ] += 1

    return matrix, label_array

def precision(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification precision."""
    if not isinstance(average, str):
        raise TypeError("average must be a string.")

    valid_averages = (
        "binary",
        "macro",
        "micro",
        "weighted",
    )

    if average not in valid_averages:
        raise ValueError(
            "average must be 'binary', 'macro', "
            "'micro' or 'weighted'."
        )

    if isinstance(zero_division, (bool, np.bool_)):
        raise TypeError("zero_division must be 0 or 1.")

    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1.")

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = np.sum(
        matrix,
        axis=0,
        dtype=np.float64,
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        true_positive,
        predicted_positive,
        out=per_class,
        where=predicted_positive != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        denominator = float(np.sum(predicted_positive))

        if denominator == 0:
            return float(zero_division)

        return float(
            np.sum(true_positive) / denominator
        )

    support = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    return float(
        np.average(per_class, weights=support)
    )

def recall(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification recall."""
    if not isinstance(average, str):
        raise TypeError("average must be a string.")

    valid_averages = (
        "binary",
        "macro",
        "micro",
        "weighted",
    )

    if average not in valid_averages:
        raise ValueError(
            "average must be 'binary', 'macro', "
            "'micro' or 'weighted'."
        )

    if isinstance(zero_division, (bool, np.bool_)):
        raise TypeError("zero_division must be 0 or 1.")

    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1.")

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)
    actual_positive = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        true_positive,
        actual_positive,
        out=per_class,
        where=actual_positive != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        denominator = float(np.sum(actual_positive))

        if denominator == 0:
            return float(zero_division)

        return float(
            np.sum(true_positive) / denominator
        )

    return float(
        np.average(per_class, weights=actual_positive)
    )

def f1(
    y_true,
    y_pred,
    average="binary",
    pos_label=1,
    zero_division=0,
):
    """Return binary or averaged classification F1 score."""
    if not isinstance(average, str):
        raise TypeError("average must be a string.")

    valid_averages = (
        "binary",
        "macro",
        "micro",
        "weighted",
    )

    if average not in valid_averages:
        raise ValueError(
            "average must be 'binary', 'macro', "
            "'micro' or 'weighted'."
        )

    if isinstance(zero_division, (bool, np.bool_)):
        raise TypeError("zero_division must be 0 or 1.")

    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1.")

    matrix, labels = confusion_matrix(y_true, y_pred)

    true_positive = np.diag(matrix).astype(np.float64)

    false_positive = (
        np.sum(matrix, axis=0, dtype=np.float64)
        - true_positive
    )

    false_negative = (
        np.sum(matrix, axis=1, dtype=np.float64)
        - true_positive
    )

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    per_class = np.full(
        true_positive.shape,
        float(zero_division),
        dtype=np.float64,
    )

    np.divide(
        2 * true_positive,
        denominator,
        out=per_class,
        where=denominator != 0,
    )

    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                "Binary averaging supports at most two classes."
            )

        matches = np.flatnonzero(labels == pos_label)

        if matches.size == 0:
            raise ValueError(
                "pos_label must be an observed class."
            )

        return float(per_class[int(matches[0])])

    if average == "macro":
        return float(np.mean(per_class))

    if average == "micro":
        numerator = 2 * float(np.sum(true_positive))
        micro_denominator = (
            numerator
            + float(np.sum(false_positive))
            + float(np.sum(false_negative))
        )

        if micro_denominator == 0:
            return float(zero_division)

        return float(numerator / micro_denominator)

    support = np.sum(
        matrix,
        axis=1,
        dtype=np.float64,
    )

    return float(
        np.average(per_class, weights=support)
    )

def mse(y_true, y_pred):
    """Return the mean squared error over all elements."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim not in (1, 2):
        raise ValueError("y_true must be 1D or 2D.")

    if pred.ndim not in (1, 2):
        raise ValueError("y_pred must be 1D or 2D.")

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes."
        )

    if true.size == 0:
        raise ValueError(
            "y_true and y_pred must not be empty."
        )

    if true.dtype.kind not in "iuf":
        raise TypeError("y_true must contain real numbers.")

    if pred.dtype.kind not in "iuf":
        raise TypeError("y_pred must contain real numbers.")

    true = true.astype(np.float64, copy=True)
    pred = pred.astype(np.float64, copy=True)

    if not np.all(np.isfinite(true)):
        raise ValueError(
            "y_true must contain only finite values."
        )

    if not np.all(np.isfinite(pred)):
        raise ValueError(
            "y_pred must contain only finite values."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        squared_errors = (true - pred) ** 2
        result = np.mean(
            squared_errors,
            dtype=np.float64,
        )

    if not np.isfinite(result):
        raise ValueError(
            "Mean squared error exceeds the numeric range."
        )

    return float(result)