"""Tests for sample-based rolling classification windows."""

import unittest
import numpy as np
from numcompute_stream.metrics import (
    RollingClassificationMetrics,
    StreamingClassificationMetrics,
)


class RollingClassificationMetricsTests(unittest.TestCase):

    def test_partial_eviction_across_chunk_boundary(self):
        metric = RollingClassificationMetrics(5)
        self.assertIs(metric.update([0, 1, 0], [0, 0, 0]), metric)
        metric.update([1, 1, 0, 1], [1, 0, 1, 1])

        result = metric.result()
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[1, 1], [1, 2]],
        )
        self.assertAlmostEqual(result["accuracy"], 3/5)
        self.assertEqual(result["n_samples"], 5)
        self.assertEqual(result["n_samples_seen"], 7)

    def test_oversized_chunk_retains_only_tail(self):
        metric = RollingClassificationMetrics(3)
        metric.update([9], [9])
        metric.update([0, 0, 1, 1, 0], [1, 0, 1, 0, 0])
        result = metric.result()

        self.assertEqual(result["classes"].tolist(), [9, 0, 1])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        )
        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(metric.n_samples_seen_, 6)

    def test_every_update_matches_latest_sample_reference(self):
        truth = np.array([0, 1, 2, 1, 0, 2, 2, 0, 1])
        prediction = np.array([0, 0, 2, 2, 0, 1, 2, 1, 1])
        metric = RollingClassificationMetrics(4)

        for start, end in ((0, 2), (2, 5), (5, 6), (6, 9)):
            metric.update(truth[start:end], prediction[start:end])
            reference = StreamingClassificationMetrics(classes=metric.classes_)
            reference.update(
                truth[max(0, end - 4):end],
                prediction[max(0, end - 4):end],
            )
            actual = metric.result()
            expected = reference.result()

            for name in (
                "confusion_matrix", "support", "predicted_support",
                "accuracy", "precision", "recall", "f1",
                "precision_macro", "recall_macro", "f1_macro",
                "precision_weighted", "recall_weighted", "f1_weighted",
            ):
                with self.subTest(end=end, metric=name):
                    np.testing.assert_allclose(actual[name], expected[name])

    def test_departed_classes_keep_order_and_macro_membership(self):
        metric = RollingClassificationMetrics(1)
        metric.update(["b"], ["b"])
        metric.update(["a"], ["a"])
        result = metric.result()

        self.assertEqual(result["classes"].tolist(), ["b", "a"])
        np.testing.assert_array_equal(result["support"], [0, 1])
        np.testing.assert_allclose(result["f1"], [0, 1])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["f1_macro"], 0.5)
        self.assertEqual(result["f1_weighted"], 1)

    def test_invalid_full_chunk_preserves_window(self):
        metric = RollingClassificationMetrics(2)
        metric.update([0, 1], [0, 1])
        before = metric.result()

        cases = [
            ([], [], ValueError),
            ([0], [0, 1], ValueError),
            ([None, 0, 1], [0, 0, 1], TypeError),
            (["x", 0, 1], [0, 0, 1], TypeError),
            (["x"], ["x"], TypeError),
        ]
        for truth, prediction, error in cases:
            with self.subTest(truth=truth, prediction=prediction):
                with self.assertRaises(error):
                    metric.update(truth, prediction)
                after = metric.result()
                np.testing.assert_array_equal(
                    after["classes"], before["classes"]
                )
                np.testing.assert_array_equal(
                    after["confusion_matrix"], before["confusion_matrix"]
                )
                self.assertEqual(after["n_samples_seen"], 2)
                self.assertEqual(after["accuracy"], 1)

    def test_input_and_result_arrays_are_independent(self):
        truth = np.array([0, 1])
        prediction = np.array([0, 1])
        metric = RollingClassificationMetrics(3)
        metric.update(truth, prediction)

        truth[:] = 9
        prediction[:] = 9
        exposed = metric.result()
        exposed["classes"][:] = 99
        exposed["confusion_matrix"][:] = 99
        exposed["f1"][:] = 99

        metric.update([0], [1])
        result = metric.result()
        self.assertEqual(result["classes"].tolist(), [0, 1])
        np.testing.assert_array_equal(
            result["confusion_matrix"], [[1, 1], [0, 1]]
        )
        self.assertAlmostEqual(result["accuracy"], 2/3)

    def test_reset_retains_configuration_and_initial_classes(self):
        initial_classes = np.array([5, 2])
        metric = RollingClassificationMetrics(
            3, classes=initial_classes, zero_division=1
        )
        initial_classes[:] = 99
        metric.update([1, 5], [2, 5])

        self.assertIs(metric.reset(), metric)
        self.assertEqual(metric.window_size, 3)
        self.assertEqual(metric.zero_division, 1)
        self.assertEqual(metric.classes_.tolist(), [5, 2])
        self.assertEqual(metric.n_samples_seen_, 0)
        self.assertEqual(metric.result()["n_samples"], 0)
        self.assertTrue(np.isnan(metric.result()["accuracy"]))

        metric.update([5], [5])
        np.testing.assert_allclose(metric.result()["f1"], [1, 1])

        dynamic = RollingClassificationMetrics(2)
        dynamic.update([0], [0])
        dynamic.reset()
        dynamic.update(["cat"], ["cat"])
        self.assertEqual(dynamic.classes_.tolist(), ["cat"])

    def test_invalid_configuration_and_package_export(self):
        for size in (True, 2.5, "3"):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    RollingClassificationMetrics(size)
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    RollingClassificationMetrics(size)

        with self.assertRaises(ValueError):
            RollingClassificationMetrics(3, zero_division=2)
        with self.assertRaises(ValueError):
            RollingClassificationMetrics(3, classes=[1, 1])

        from numcompute_stream import (
            RollingClassificationMetrics as ExportedMetrics,
        )
        self.assertIs(ExportedMetrics, RollingClassificationMetrics)


if __name__ == "__main__":
    unittest.main()