"""Tests for cumulative confusion counts and stable label mapping."""

import unittest
import numpy as np
from numcompute_stream.metrics import StreamingConfusionMatrix


class StreamingConfusionMatrixTests(unittest.TestCase):

    def test_matrix_orientation_and_unequal_chunks(self):
        metric = StreamingConfusionMatrix()

        self.assertIs(metric.update([0, 1, 1], [0, 0, 1]), metric)
        metric.update([1], [0])

        result = metric.result()
        np.testing.assert_array_equal(result["classes"], [0, 1])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[1, 0], [2, 1]],
        )
        self.assertEqual(result["n_samples"], 4)

        batch = StreamingConfusionMatrix()
        batch.update([0, 1, 1, 1], [0, 0, 1, 0])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            batch.result()["confusion_matrix"],
        )

    def test_new_classes_append_without_reordering(self):
        metric = StreamingConfusionMatrix()
        metric.update([2, 5], [5, 5])
        metric.update([1, 5], [2, 1])

        result = metric.result()
        np.testing.assert_array_equal(result["classes"], [2, 5, 1])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[0, 1, 0], [0, 1, 1], [1, 0, 0]],
        )

    def test_prediction_only_class_and_string_labels(self):
        metric = StreamingConfusionMatrix()
        metric.update(["cat", "cat"], ["dog", "cat"])

        result = metric.result()
        np.testing.assert_array_equal(result["classes"], ["cat", "dog"])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[1, 1], [0, 0]],
        )

    def test_large_signed_and_unsigned_integer_labels(self):
        large = 2**63 + 7
        metric = StreamingConfusionMatrix()
        metric.update(
            np.array([-1], dtype=np.int64),
            np.array([large], dtype=np.uint64),
        )
        metric.update(
            np.array([large], dtype=np.uint64),
            np.array([-1], dtype=np.int64),
        )

        result = metric.result()
        self.assertEqual(result["classes"].tolist(), [-1, large])
        np.testing.assert_array_equal(
            result["confusion_matrix"],
            [[0, 1], [1, 0]],
        )

    def test_invalid_updates_preserve_state(self):
        metric = StreamingConfusionMatrix()
        metric.update([0, 1], [0, 1])
        before = metric.result()

        cases = [
            ([0], [0, 1], ValueError),
            ([], [], ValueError),
            ([[0]], [0], ValueError),
            ([0.0], [0], TypeError),
            ([True], [0], TypeError),
            ([None], [0], TypeError),
            ([0, "cat"], [0, 1], TypeError),
            ([0], ["cat"], TypeError),
            (["cat"], ["cat"], TypeError),
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
                self.assertEqual(after["n_samples"], before["n_samples"])

    def test_reset_retains_initial_order_and_copies(self):
        initial = np.array([5, 2])
        metric = StreamingConfusionMatrix(classes=initial)
        initial[:] = 99

        metric.update([1, 5], [2, 1])
        result = metric.result()
        result["classes"][:] = 99
        result["confusion_matrix"][:] = 99
        exposed_classes = metric.classes_
        exposed_classes[:] = 99

        self.assertEqual(metric.classes_.tolist(), [5, 2, 1])
        self.assertEqual(metric.result()["confusion_matrix"].sum(), 2)

        self.assertIs(metric.reset(), metric)
        self.assertEqual(metric.classes_.tolist(), [5, 2])
        self.assertEqual(metric.n_samples_seen_, 0)
        np.testing.assert_array_equal(
            metric.result()["confusion_matrix"],
            np.zeros((2, 2), dtype=np.int64),
        )

        dynamic = StreamingConfusionMatrix()
        dynamic.update([1], [1])
        dynamic.reset()
        self.assertEqual(dynamic.result()["confusion_matrix"].shape, (0, 0))
        dynamic.update(["cat"], ["cat"])
        self.assertEqual(dynamic.classes_.tolist(), ["cat"])

    def test_invalid_initial_classes(self):
        with self.assertRaises(ValueError):
            StreamingConfusionMatrix(classes=[1, 1])
        with self.assertRaises(ValueError):
            StreamingConfusionMatrix(classes=[])
        with self.assertRaises(TypeError):
            StreamingConfusionMatrix(classes=[1, "cat"])


if __name__ == "__main__":
    unittest.main()