"""Tests for comparison plots with aligned streaming metrics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.visualise import compare_models


class TestCompareModels(unittest.TestCase):
    def setUp(self):
        self.initial_figures = set(plt.get_fignums())
        self.addCleanup(self.close_created_figures)

    def close_created_figures(self):
        for number in set(plt.get_fignums()) - self.initial_figures:
            plt.close(number)

    def test_default_coordinates_labels_and_values(self):
        fig, ax = compare_models([0.5, 0.7], [0.6, 0.8], ["First", "Second"])
        self.assertIs(ax.figure, fig)
        self.assertEqual(len(ax.lines), 2)
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), [1, 2])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.5, 0.7])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.6, 0.8])
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["First", "Second"],
        )
        self.assertEqual(ax.get_xlabel(), "Chunk")
        self.assertEqual(ax.get_ylabel(), "Metric")
        fig.canvas.draw()

    def test_different_nan_gaps_keep_shared_coordinates(self):
        first = np.array([np.nan, 0.7, 0.8])
        second = np.array([np.nan, np.nan, 0.9])
        coordinates = np.array([2, 5, 9])
        originals = [array.copy() for array in (
            first, second, coordinates
        )]
        for array in (first, second, coordinates):
            array.setflags(write=False)

        _, ax = compare_models(
            first,
            second,
            ["A", "B"],
            x_values=coordinates,
        )

        for line, expected in zip(ax.lines, originals[:2]):
            np.testing.assert_array_equal(line.get_xdata(), originals[2])
            np.testing.assert_allclose(line.get_ydata(), expected, equal_nan=True)
        for actual, expected in zip((first, second, coordinates), originals):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_all_nan_series_and_underscore_labels(self):
        fig, ax = compare_models(
            [np.nan, np.nan],
            [np.nan, np.nan],
            [" _first ", "_second"],
        )
        fig.canvas.draw()
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["_first", "_second"],
        )
        for line in ax.lines:
            self.assertTrue(np.isnan(line.get_ydata()).all())

    def test_existing_axes_preserve_original_line(self):
        fig, ax = plt.subplots()
        original, = ax.plot([1, 2], [0.1, 0.2], label="Existing")
        before = set(plt.get_fignums())

        returned_fig, returned_ax = compare_models(
            [0.5, 0.7],
            [0.6, 0.8],
            ["A", "B"],
            ax=ax,
            title="Accuracy comparison",
            ylabel="Accuracy",
        )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertIs(ax.lines[0], original)
        self.assertEqual(len(ax.lines), 3)
        self.assertEqual(set(plt.get_fignums()), before)
        self.assertEqual(ax.get_title(), "Accuracy comparison")
        self.assertEqual(ax.get_ylabel(), "Accuracy")
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["A", "B"],
        )

    def test_invalid_series_do_not_modify_existing_axes(self):
        fig, ax = plt.subplots()
        original, = ax.plot([1], [0.2])
        cases = [
            ([0.5], [0.6, 0.7], ValueError),
            ([], [], ValueError),
            ([0.5], [np.inf], ValueError),
            ([0.5], [[0.6]], ValueError),
            ([0.5], ["0.6"], TypeError),
            ([True], [0.6], TypeError),
        ]
        for first, second, error in cases:
            with self.subTest(first=first, second=second):
                with self.assertRaises(error):
                    compare_models(first, second, ["A", "B"], ax=ax)
                self.assertEqual(len(ax.lines), 1)
                self.assertIs(ax.lines[0], original)

    def test_invalid_labels_create_no_figures(self):
        cases = [
            ("AB", TypeError),
            (None, TypeError),
            (["A"], ValueError),
            (["A", "B", "C"], ValueError),
            (["A", 2], TypeError),
            (["A", " "], ValueError),
            (["A", "A"], ValueError),
            ([" A ", "A"], ValueError),
        ]
        before = set(plt.get_fignums())
        for labels, error in cases:
            with self.subTest(labels=labels):
                with self.assertRaises(error):
                    compare_models([0.5], [0.6], labels)
                self.assertEqual(set(plt.get_fignums()), before)

    def test_invalid_options_create_no_figures(self):
        cases = [
            ({"x_values": [1]}, ValueError),
            ({"x_values": [2, 1]}, ValueError),
            ({"x_values": [1, np.nan]}, ValueError),
            ({"title": None}, TypeError),
            ({"ylabel": 1}, TypeError),
            ({"show": "yes"}, TypeError),
            ({"ax": object()}, TypeError),
        ]
        before = set(plt.get_fignums())
        for options, error in cases:
            with self.subTest(options=options):
                with self.assertRaises(error):
                    compare_models(
                        [0.5, 0.7], [0.6, 0.8], ["A", "B"], **options
                    )
                self.assertEqual(set(plt.get_fignums()), before)

    def test_save_and_show_include_both_lines(self):
        observed = []

        def inspect_show():
            ax = plt.gcf().axes[0]
            observed.append((
                len(ax.lines),
                [text.get_text()
                 for text in ax.get_legend().get_texts()],
            ))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            with patch.object(
                plt, "show", side_effect=inspect_show
            ) as mocked_show:
                fig, _ = compare_models(
                    [0.5, 0.7],
                    [0.6, 0.8],
                    ["A", "B"],
                    save_path=output,
                    show=True,
                )
                mocked_show.assert_called_once_with()

            self.assertEqual(observed, [(2, ["A", "B"])])
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(plt.fignum_exists(fig.number))

        with patch.object(plt, "show") as mocked_show:
            compare_models([0.5], [0.6], ["A", "B"])
            mocked_show.assert_not_called()

    def test_invalid_output_path_does_not_modify_axes(self):
        fig, ax = plt.subplots()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (root / "missing" / "plot.png", FileNotFoundError),
                (root, IsADirectoryError),
            ]
            for path, error in cases:
                with self.subTest(path=path):
                    with self.assertRaises(error):
                        compare_models(
                            [0.5], [0.6], ["A", "B"],
                            ax=ax, save_path=path,
                        )
                    self.assertEqual(len(ax.lines), 0)

    def test_save_failure_respects_figure_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            before = set(plt.get_fignums())

            with patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=OSError("simulated save failure"),
            ):
                with self.assertRaises(OSError):
                    compare_models([0.5], [0.6], ["A", "B"], save_path=output)
            self.assertEqual(set(plt.get_fignums()), before)

            fig, ax = plt.subplots()
            with patch.object(fig, "savefig", side_effect=OSError("save failed")):
                with self.assertRaises(OSError):
                    compare_models(
                        [0.5], [0.6], ["A", "B"],
                        ax=ax, save_path=output,
                    )
            self.assertTrue(plt.fignum_exists(fig.number))

    def test_second_line_failure_closes_new_figure(self):
        from matplotlib.axes import Axes

        original_plot = Axes.plot
        call_count = 0

        def fail_second_plot(axes, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated second curve failure")
            return original_plot(axes, *args, **kwargs)

        before = set(plt.get_fignums())
        with patch.object(Axes, "plot", new=fail_second_plot):
            with self.assertRaisesRegex(RuntimeError, "second curve failure"):
                compare_models([0.5], [0.6], ["A", "B"])

        self.assertEqual(call_count, 2)
        self.assertEqual(set(plt.get_fignums()), before)


if __name__ == "__main__":
    unittest.main()