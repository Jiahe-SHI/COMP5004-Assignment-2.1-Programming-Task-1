"""Tests for classification metrics derived from cumulative counts."""

import unittest
import numpy as np
from numcompute_stream.metrics import StreamingClassificationMetrics


class StreamingClassificationMetricsTests(unittest.TestCase):

    def test_hand_calculated_multiclass_metrics(self):
        metric = StreamingClassificationMetrics(classes=[0, 1, 2])
        metric.update(
            [0, 0, 0, 1, 1, 2],
            [0, 0, 1, 1, 2, 2],
        )
        result = metric.result()

        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[2, 1, 0], [0, 1, 1], [0, 0, 1]],
        )
        np.testing.assert_array_equal(result["support"], [3, 2, 1])
        np.testing.assert_array_equal(
            result["predicted_support"], [2, 2, 2]
        )
        np.testing.assert_allclose(result["precision"], [1, 1/2, 1/2])
        np.testing.assert_allclose(result["recall"], [2/3, 1/2, 1])
        np.testing.assert_allclose(result["f1"], [4/5, 1/2, 2/3])

        self.assertAlmostEqual(result["accuracy"], 2/3)
        self.assertAlmostEqual(result["precision_macro"], 2/3)
        self.assertAlmostEqual(result["recall_macro"], 13/18)
        self.assertAlmostEqual(result["f1_macro"], 59/90)
        self.assertAlmostEqual(result["precision_weighted"], 3/4)
        self.assertAlmostEqual(result["recall_weighted"], 2/3)
        self.assertAlmostEqual(result["f1_weighted"], 61/90)

    def test_unequal_chunks_use_total_counts(self):
        metric = StreamingClassificationMetrics()
        metric.update([0], [0])
        metric.update([1, 1, 1], [0, 0, 0])

        result = metric.result()
        self.assertAlmostEqual(result["accuracy"], 1/4)
        self.assertNotEqual(result["accuracy"], 1/2)

        batch = StreamingClassificationMetrics()
        batch.update([0, 1, 1, 1], [0, 0, 0, 0])
        expected = batch.result()
        for name in (
            "precision", "recall", "f1",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ):
            with self.subTest(metric=name):
                np.testing.assert_allclose(result[name], expected[name])

    def test_zero_denominators_and_absent_class(self):
        for fallback in (0, 1):
            with self.subTest(zero_division=fallback):
                metric = StreamingClassificationMetrics(
                    classes=[0, 1, 2], zero_division=fallback
                )
                metric.update([0], [1])
                result = metric.result()

                np.testing.assert_allclose(
                    result["precision"], [fallback, 0, fallback]
                )
                np.testing.assert_allclose(
                    result["recall"], [0, fallback, fallback]
                )
                np.testing.assert_allclose(
                    result["f1"], [0, 0, fallback]
                )
                self.assertEqual(result["accuracy"], 0)
                self.assertAlmostEqual(result["f1_macro"], fallback / 3)
                self.assertEqual(result["f1_weighted"], 0)

    def test_no_samples_returns_nan_without_warnings(self):
        for classes in (None, [0, 1]):
            for fallback in (0, 1):
                with self.subTest(classes=classes, fallback=fallback):
                    metric = StreamingClassificationMetrics(
                        classes=classes, zero_division=fallback
                    )
                    result = metric.result()
                    self.assertEqual(result["n_samples"], 0)
                    self.assertTrue(np.isnan(result["accuracy"]))
                    for name in ("precision", "recall", "f1"):
                        self.assertTrue(np.isnan(result[name]).all())
                        self.assertTrue(np.isnan(result[f"{name}_macro"]))
                        self.assertTrue(np.isnan(result[f"{name}_weighted"]))

    def test_dynamic_class_order_matches_metric_arrays(self):
        metric = StreamingClassificationMetrics()
        metric.update(["b", "c"], ["b", "b"])
        metric.update(["a"], ["c"])
        result = metric.result()

        self.assertEqual(result["classes"].tolist(), ["b", "c", "a"])
        np.testing.assert_array_equal(result["support"], [1, 1, 1])
        np.testing.assert_allclose(result["precision"], [1/2, 0, 0])
        np.testing.assert_allclose(result["recall"], [1, 0, 0])
        np.testing.assert_allclose(result["f1"], [2/3, 0, 0])

    def test_query_ownership_and_reset_configuration(self):
        metric = StreamingClassificationMetrics(
            classes=[1, 0], zero_division=1
        )
        self.assertIs(metric.update([1, 0], [1, 1]), metric)

        first = metric.result()
        for value in first.values():
            if isinstance(value, np.ndarray):
                value[:] = 99

        second = metric.result()
        self.assertEqual(second["n_samples"], 2)
        self.assertAlmostEqual(second["accuracy"], 1/2)
        np.testing.assert_allclose(second["precision"], [1/2, 1])
        np.testing.assert_array_equal(second["support"], [1, 1])

        self.assertIs(metric.reset(), metric)
        self.assertEqual(metric.classes_.tolist(), [1, 0])
        self.assertEqual(metric.zero_division, 1)
        self.assertTrue(np.isnan(metric.result()["accuracy"]))

        metric.update([1], [1])
        np.testing.assert_allclose(metric.result()["f1"], [1, 1])

    def test_configuration_validation_and_package_exports(self):
        for value in (True, "0", None, [0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    StreamingClassificationMetrics(zero_division=value)

        for value in (-1, 2, 0.5, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    StreamingClassificationMetrics(zero_division=value)

        from numcompute_stream import (
            StreamingClassificationMetrics as ExportedMetrics,
            StreamingConfusionMatrix as ExportedMatrix,
        )
        from numcompute_stream.metrics import StreamingConfusionMatrix

        self.assertIs(ExportedMetrics, StreamingClassificationMetrics)
        self.assertIs(ExportedMatrix, StreamingConfusionMatrix)


if __name__ == "__main__":
    unittest.main()