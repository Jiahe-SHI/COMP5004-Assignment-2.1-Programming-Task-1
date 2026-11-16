"""Tests for categorical prediction plots."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream.visualise import (
    plot_predictions_vs_ground_truth,
)


class TestPredictionPlot(unittest.TestCase):
    def setUp(self):
        self.initial_figures = set(plt.get_fignums())
        self.addCleanup(self.close_created_figures)

    def close_created_figures(self):
        for number in set(plt.get_fignums()) - self.initial_figures:
            plt.close(number)

    def test_integer_labels_use_shared_categorical_positions(self):
        fig, ax = plot_predictions_vs_ground_truth(
            [10, 30, 10], [30, 30, 20]
        )
        self.assertIs(ax.figure, fig)
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["10", "20", "30"],
        )
        true_points = np.asarray(ax.collections[0].get_offsets())
        predicted_points = np.asarray(ax.collections[1].get_offsets())
        np.testing.assert_array_equal(true_points, [[1, 0], [2, 2], [3, 0]])
        np.testing.assert_array_equal(predicted_points, [[1, 2], [2, 2], [3, 1]])
        self.assertEqual(ax.get_xlabel(), "Sample")
        self.assertEqual(ax.get_ylabel(), "Class")
        fig.canvas.draw()

    def test_string_labels_and_explicit_positions_preserve_inputs(self):
        truth = np.array(["cat", "dog", "cat"])
        prediction = np.array(["fox", "dog", "cat"])
        indices = np.array([385, 390, 400])
        originals = [array.copy() for array in (truth, prediction, indices)]
        for array in (truth, prediction, indices):
            array.setflags(write=False)

        _, ax = plot_predictions_vs_ground_truth(
            truth, prediction, sample_indices=indices
        )

        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["cat", "dog", "fox"],
        )
        for artist in ax.collections:
            np.testing.assert_array_equal(
                np.asarray(artist.get_offsets())[:, 0], indices
            )
        for actual, expected in zip(
            (truth, prediction, indices), originals
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_large_integer_labels_remain_distinct(self):
        large = 2**64
        truth = np.array([large, large + 1], dtype=object)
        prediction = np.array([large + 1, large], dtype=object)

        _, ax = plot_predictions_vs_ground_truth(truth, prediction)

        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            [str(large), str(large + 1)],
        )
        np.testing.assert_array_equal(
            np.asarray(ax.collections[0].get_offsets())[:, 1],
            [0, 1],
        )
        np.testing.assert_array_equal(
            np.asarray(ax.collections[1].get_offsets())[:, 1],
            [1, 0],
        )

    def test_single_sample_single_class_can_render(self):
        fig, ax = plot_predictions_vs_ground_truth(["yes"], ["yes"])
        fig.canvas.draw()
        self.assertEqual(
            [tick.get_text() for tick in ax.get_yticklabels()],
            ["yes"],
        )
        np.testing.assert_allclose(ax.get_ylim(), [-0.5, 0.5])
        for artist in ax.collections:
            np.testing.assert_array_equal(np.asarray(artist.get_offsets()), [[1, 0]])

    def test_invalid_labels_create_no_figures(self):
        cases = [
            ([], []),
            ([0, 1], [0]),
            ([[0, 1]], [0, 1]),
            ([0, 1], [[0, 1]]),
            ([0.0], [0]),
            ([True], [0]),
            ([None], [0]),
            ([0], [np.nan]),
            ([0], ["zero"]),
            (np.array([0, "one"], dtype=object), [0, 1]),
        ]
        before = set(plt.get_fignums())
        for truth, prediction in cases:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises((TypeError, ValueError)):
                    plot_predictions_vs_ground_truth(truth, prediction)
                self.assertEqual(set(plt.get_fignums()), before)

    def test_invalid_positions_do_not_modify_existing_axes(self):
        fig, ax = plt.subplots()
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
        for positions, error in cases:
            with self.subTest(positions=positions):
                with self.assertRaises(error):
                    plot_predictions_vs_ground_truth(
                        [0, 1],
                        [1, 0],
                        sample_indices=positions,
                        ax=ax,
                    )
                self.assertEqual(len(ax.collections), 0)

    def test_existing_axes_are_reused_without_clearing(self):
        fig, ax = plt.subplots()
        original = ax.scatter([1], [0])
        before = set(plt.get_fignums())

        returned_fig, returned_ax = plot_predictions_vs_ground_truth(
            [0, 1], [1, 1], ax=ax, title="Prediction check"
        )

        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        self.assertIs(ax.collections[0], original)
        self.assertEqual(len(ax.collections), 3)
        self.assertEqual(set(plt.get_fignums()), before)
        self.assertEqual(ax.get_title(), "Prediction check")
        self.assertEqual(
            [text.get_text() for text in ax.get_legend().get_texts()],
            ["Ground truth", "Prediction"],
        )

    def test_png_saving_and_explicit_show(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.png"
            with patch.object(plt, "show") as mocked_show:
                fig, _ = plot_predictions_vs_ground_truth(
                    [0, 1], [1, 1], save_path=output
                )
                mocked_show.assert_not_called()

            self.assertEqual(
                output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )
            self.assertTrue(plt.fignum_exists(fig.number))

        with patch.object(plt, "show") as mocked_show:
            plot_predictions_vs_ground_truth([0], [0], show=True)
            mocked_show.assert_called_once_with()

    def test_invalid_options_and_paths_create_no_figures(self):
        before = set(plt.get_fignums())
        for options in (
            {"title": None},
            {"show": "yes"},
            {"ax": object()},
        ):
            with self.subTest(options=options):
                with self.assertRaises(TypeError):
                    plot_predictions_vs_ground_truth(
                        [0], [0], **options
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                plot_predictions_vs_ground_truth(
                    [0], [0],
                    save_path=root / "missing" / "predictions.png",
                )
            with self.assertRaises(IsADirectoryError):
                plot_predictions_vs_ground_truth(
                    [0], [0], save_path=root
                )

        self.assertEqual(set(plt.get_fignums()), before)

    def test_save_failure_respects_figure_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.png"
            before = set(plt.get_fignums())

            with patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=OSError("simulated save failure"),
            ):
                with self.assertRaises(OSError):
                    plot_predictions_vs_ground_truth(
                        [0], [0], save_path=output
                    )
            self.assertEqual(set(plt.get_fignums()), before)

            fig, ax = plt.subplots()
            with patch.object(
                fig, "savefig", side_effect=OSError("save failed")
            ):
                with self.assertRaises(OSError):
                    plot_predictions_vs_ground_truth(
                        [0], [0], ax=ax, save_path=output
                    )
            self.assertTrue(plt.fignum_exists(fig.number))

    def test_second_scatter_failure_closes_new_figure(self):
        from matplotlib.axes import Axes

        original_scatter = Axes.scatter
        call_count = 0

        def fail_second_scatter(axes, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated prediction scatter failure")
            return original_scatter(axes, *args, **kwargs)

        before = set(plt.get_fignums())
        with patch.object(Axes, "scatter", new=fail_second_scatter):
            with self.assertRaisesRegex(RuntimeError, "prediction scatter failure"):
                plot_predictions_vs_ground_truth([0, 1], [1, 0])

        self.assertEqual(call_count, 2)
        self.assertEqual(set(plt.get_fignums()), before)


if __name__ == "__main__":
    unittest.main()