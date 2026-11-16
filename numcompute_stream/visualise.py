"""Matplotlib visualisations of streaming results."""

from pathlib import Path

import numpy as np


def _numeric_series(values, name, *, allow_nan):
    """Return an independent, non-empty 1-D real numeric array."""
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")

    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinity")
    if not allow_nan and np.isnan(array).any():
        raise ValueError(f"{name} must contain only finite values")

    return array.copy()


def plot_metric_over_time(
    metric_values,
    title,
    ylabel,
    *,
    x_values=None,
    ax=None,
    save_path=None,
    show=False,
):
    """Plot a metric against chunk numbers and return (fig, ax).

    metric_values must be a non-empty 1-D real numeric sequence.
    NaN values are preserved as gaps, including all-NaN warm-up data.
    Infinity, boolean, complex and non-numeric values are rejected.

    x_values defaults to 1, ..., n. Explicit coordinates must be finite,
    strictly increasing and the same length as metric_values.

    If ax is supplied, append a line to that axes without clearing it.
    Otherwise create a new figure. The caller owns the returned figure
    and should close it when finished.

    save_path optionally saves the figure; its parent directory must
    already exist. show=True calls matplotlib.pyplot.show().
    This function does not select a Matplotlib backend.
    """
    values = _numeric_series(
        metric_values, "metric_values", allow_nan=True
    )
    if not isinstance(title, str) or not isinstance(ylabel, str):
        raise TypeError("title and ylabel must be strings")
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    if x_values is None:
        coordinates = np.arange(1, values.size + 1)
    else:
        coordinates = _numeric_series(
            x_values, "x_values", allow_nan=False
        )
        if coordinates.size != values.size:
            raise ValueError(
                "x_values and metric_values must have the same length"
            )
        if not np.all(coordinates[1:] > coordinates[:-1]):
            raise ValueError("x_values must be strictly increasing")

    destination = None
    if save_path is not None:
        destination = Path(save_path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {destination.parent}"
            )
        if destination.is_dir():
            raise IsADirectoryError(str(destination))

    # Import lazily so importing this module does not initialise pyplot.
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes or None")

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    try:
        ax.plot(coordinates, values, marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Chunk")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if created:
            fig.tight_layout()
        if destination is not None:
            fig.savefig(destination, bbox_inches="tight")
        if show:
            plt.show()
    except Exception:
        if created:
            plt.close(fig)
        raise

    return fig, ax

def compare_models(
    metric1,
    metric2,
    labels,
    *,
    x_values=None,
    title="Model comparison",
    ylabel="Metric",
    ax=None,
    save_path=None,
    show=False,
):
    """Compare two metric series at shared chunk coordinates.

    Both series must be non-empty, one-dimensional and equal in length.
    Preserve unavailable measurements as NaN at their original positions.
    The caller must supply the same metric and evaluation meaning for
    both series; this function cannot verify those semantics.

    labels must contain two distinct, non-empty strings. Labels are
    passed explicitly to the legend, including names starting with "_".

    Coordinates, output paths and figure ownership follow
    plot_metric_over_time. Existing axes are not cleared. The legend
    describes the two newly added comparison lines.

    Return (fig, ax). A newly created figure is closed on failure;
    changes to caller-supplied axes are not rolled back.
    """
    first = _numeric_series(metric1, "metric1", allow_nan=True)
    second = _numeric_series(metric2, "metric2", allow_nan=True)
    if first.size != second.size:
        raise ValueError("metric1 and metric2 must have the same length")

    if isinstance(labels, (str, bytes)):
        raise TypeError("labels must contain two strings")
    try:
        names = list(labels)
    except TypeError as exc:
        raise TypeError("labels must contain two strings") from exc

    if len(names) != 2:
        raise ValueError("labels must contain exactly two names")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("labels must contain only strings")
    names = [name.strip() for name in names]
    if any(not name for name in names):
        raise ValueError("labels must not be empty")
    if names[0] == names[1]:
        raise ValueError("labels must be distinct")

    # Validate options before creating a figure or modifying existing axes.
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    destination = None
    if save_path is not None:
        destination = Path(save_path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {destination.parent}"
            )
        if destination.is_dir():
            raise IsADirectoryError(str(destination))

    import matplotlib.pyplot as plt

    created = ax is None

    # Reuse the first function's coordinate and axes validation.
    # Delay saving and showing until both curves and the legend exist.
    fig, ax = plot_metric_over_time(
        first,
        title,
        ylabel,
        x_values=x_values,
        ax=ax,
        show=False,
    )

    try:
        first_line = ax.lines[-1]
        first_line.set_label(names[0])

        second_line, = ax.plot(
            first_line.get_xdata(),
            second,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label=names[1],
        )
        ax.legend(
            handles=[first_line, second_line],
            labels=names,
        )

        if created:
            fig.tight_layout()
        if destination is not None:
            fig.savefig(destination, bbox_inches="tight")
        if show:
            plt.show()
    except Exception:
        if created:
            plt.close(fig)
        raise

    return fig, ax

def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    *,
    sample_indices=None,
    title="Predictions vs ground truth",
    ax=None,
    save_path=None,
    show=False,
):
    """Plot classification labels at shared sample positions.

    Labels follow the package contract: homogeneous integers or strings,
    with no missing labels. Both inputs must be non-empty 1-D sequences
    of equal length and the same label family.

    A sorted union of true and predicted labels defines the categorical
    y-axis. Integer labels are mapped to positions without conversion
    to floating-point label values.

    sample_indices defaults to 1, ..., n. Explicit positions must be
    finite, strictly increasing and the same length as the labels.
    Exclude warm-up samples without predictions before calling.

    Return (fig, ax). Existing axes are not cleared, but their y-axis
    ticks and limits are set for this call's category mapping. Use a
    dedicated axes unless existing artists use the same mapping.

    Output paths, display and figure ownership follow the other plotting
    functions. Changes to caller-supplied axes are not rolled back.
    """
    from ._validation import validate_y

    truth = validate_y(y_true)
    predicted = validate_y(y_pred, n_samples=truth.size)
    if truth.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    # Object concatenation avoids losing large integer label precision.
    combined = np.concatenate((
        truth.astype(object),
        predicted.astype(object),
    ))
    combined = validate_y(combined)
    classes, encoded = np.unique(combined, return_inverse=True)
    true_positions = encoded[:truth.size]
    predicted_positions = encoded[truth.size:]

    if sample_indices is None:
        coordinates = np.arange(1, truth.size + 1)
    else:
        coordinates = _numeric_series(
            sample_indices, "sample_indices", allow_nan=False
        )
        if coordinates.size != truth.size:
            raise ValueError(
                "sample_indices and labels must have the same length"
            )
        if not np.all(coordinates[1:] > coordinates[:-1]):
            raise ValueError("sample_indices must be strictly increasing")

    if not isinstance(title, str):
        raise TypeError("title must be a string")
    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    destination = None
    if save_path is not None:
        destination = Path(save_path)
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {destination.parent}"
            )
        if destination.is_dir():
            raise IsADirectoryError(str(destination))

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes or None")

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    try:
        true_artist = ax.scatter(
            coordinates,
            true_positions,
            marker="o",
            facecolors="none",
            edgecolors="tab:blue",
            s=65,
            linewidths=1.5,
            label="Ground truth",
        )
        predicted_artist = ax.scatter(
            coordinates,
            predicted_positions,
            marker="x",
            color="tab:orange",
            s=40,
            linewidths=1.5,
            label="Prediction",
        )

        ax.set_yticks(np.arange(classes.size))
        ax.set_yticklabels([str(label) for label in classes])
        ax.set_ylim(-0.5, classes.size - 0.5)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Class")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[true_artist, predicted_artist],
            labels=["Ground truth", "Prediction"],
        )

        if created:
            fig.tight_layout()
        if destination is not None:
            fig.savefig(destination, bbox_inches="tight")
        if show:
            plt.show()
    except Exception:
        if created:
            plt.close(fig)
        raise

    return fig, ax