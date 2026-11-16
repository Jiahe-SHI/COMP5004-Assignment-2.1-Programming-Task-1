"""Tests for metric plotting using a non-interactive backend."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.visualise import plot_metric_over_time


class TestMetricPlot(unittest.TestCase):
    def setUp(self):
        self.initial_figures = set(plt.get_fignums())
        self.addCleanup(self.close_created_figures)

    def close_created_figures(self):
        for number in set(plt.get_fignums()) - self.initial_figures:
            plt.close(number)

    def test_default_coordinates_and_labels(self):
        fig, ax = plot_metric_over_time(
            [0.5, 0.75, 1.0], "Accuracy over time", "Accuracy"
        )
        self.assertIs(ax.figure, fig)
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), [1, 2, 3]
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), [0.5, 0.75, 1.0]
        )
        self.assertEqual(ax.get_title(), "Accuracy over time")
        self.assertEqual(ax.get_xlabel(), "Chunk")
        self.assertEqual(ax.get_ylabel(), "Accuracy")
        fig.canvas.draw()

    def test_explicit_coordinates_preserve_nan_gaps_and_inputs(self):
        values = np.array([np.nan, 0.6, np.nan, 0.9])
        coordinates = np.array([2, 4, 7, 10])
        original_values = values.copy()
        original_coordinates = coordinates.copy()
        values.setflags(write=False)
        coordinates.setflags(write=False)

        _, ax = plot_metric_over_time(
            values, "Metric", "Score", x_values=coordinates
        )
        np.testing.assert_array_equal(
            ax.lines[0].get_xdata(), original_coordinates
        )
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), original_values, equal_nan=True
        )
        np.testing.assert_allclose(
            values, original_values, equal_nan=True
        )
        np.testing.assert_array_equal(
            coordinates, original_coordinates
        )

    def test_all_nan_and_single_point_can_render(self):
        for values in ([np.nan, np.nan], [0.75]):
            with self.subTest(values=values):
                fig, ax = plot_metric_over_time(
                    values, "Warm-up or short stream", "Score"
                )
                fig.canvas.draw()
                np.testing.assert_allclose(
                    ax.lines[0].get_ydata(), values, equal_nan=True
                )
                plt.close(fig)

    def test_existing_axes_are_reused_without_clearing(self):
        fig, ax = plt.subplots()
        original_line, = ax.plot([1, 2], [0.2, 0.3])
        figures_before = set(plt.get_fignums())

        returned_fig, returned_ax = plot_metric_over_time(
            [0.6, 0.8], "Comparison", "Score", ax=ax
        )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertEqual(len(ax.lines), 2)
        self.assertIs(ax.lines[0], original_line)
        self.assertEqual(set(plt.get_fignums()), figures_before)

    def test_png_saved_and_figure_remains_open(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metric.png"
            fig, _ = plot_metric_over_time(
                [0.5, 0.8], "Metric", "Score", save_path=output
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)
            self.assertEqual(
                output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )
            self.assertTrue(plt.fignum_exists(fig.number))

    def test_show_is_explicit(self):
        with patch.object(plt, "show") as mocked_show:
            plot_metric_over_time([0.5], "Metric", "Score")
            mocked_show.assert_not_called()

            plot_metric_over_time(
                [0.5], "Metric", "Score", show=True
            )
            mocked_show.assert_called_once_with()

    def test_invalid_metric_values_create_no_figures(self):
        cases = [
            ([], ValueError),
            ([[0.5]], ValueError),
            (0.5, ValueError),
            ([np.inf], ValueError),
            ([-np.inf], ValueError),
            ([True, False], TypeError),
            ([1 + 2j], TypeError),
            (["0.5"], TypeError),
            ([None], TypeError),
        ]
        before = set(plt.get_fignums())
        for values, error in cases:
            with self.subTest(values=values):
                with self.assertRaises(error):
                    plot_metric_over_time(values, "Metric", "Score")
                self.assertEqual(set(plt.get_fignums()), before)

    def test_invalid_coordinates_create_no_figures(self):
        cases = [
            ([], ValueError),
            ([1], ValueError),
            ([[1, 2]], ValueError),
            ([1, 1], ValueError),
            ([2, 1], ValueError),
            ([1, np.nan], ValueError),
            ([1, np.inf], ValueError),
            (["1", "2"], TypeError),
            ([True, False], TypeError),
        ]
        before = set(plt.get_fignums())
        for coordinates, error in cases:
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(error):
                    plot_metric_over_time(
                        [0.5, 0.8],
                        "Metric",
                        "Score",
                        x_values=coordinates,
                    )
                self.assertEqual(set(plt.get_fignums()), before)

    def test_invalid_options_create_no_figures(self):
        before = set(plt.get_fignums())
        for title, ylabel in ((None, "Score"), ("Metric", 1)):
            with self.subTest(title=title, ylabel=ylabel):
                with self.assertRaises(TypeError):
                    plot_metric_over_time([0.5], title, ylabel)

        with self.assertRaises(TypeError):
            plot_metric_over_time(
                [0.5], "Metric", "Score", show="yes"
            )
        with self.assertRaises(TypeError):
            plot_metric_over_time(
                [0.5], "Metric", "Score", ax=object()
            )
        self.assertEqual(set(plt.get_fignums()), before)

    def test_invalid_output_paths_create_no_figures(self):
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                plot_metric_over_time(
                    [0.5],
                    "Metric",
                    "Score",
                    save_path=root / "missing" / "metric.png",
                )
            with self.assertRaises(IsADirectoryError):
                plot_metric_over_time(
                    [0.5], "Metric", "Score", save_path=root
                )
        self.assertEqual(set(plt.get_fignums()), before)

    def test_save_failure_closes_new_figure(self):
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metric.png"
            with patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=OSError("simulated save failure"),
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated save failure"
                ):
                    plot_metric_over_time(
                        [0.5], "Metric", "Score", save_path=output
                    )
        self.assertEqual(set(plt.get_fignums()), before)

    def test_save_failure_keeps_caller_figure_open(self):
        fig, ax = plt.subplots()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metric.png"
            with patch.object(
                fig,
                "savefig",
                side_effect=OSError("simulated save failure"),
            ):
                with self.assertRaises(OSError):
                    plot_metric_over_time(
                        [0.5], "Metric", "Score", ax=ax, save_path=output,
                    )
        self.assertTrue(plt.fignum_exists(fig.number))


if __name__ == "__main__":
    unittest.main()