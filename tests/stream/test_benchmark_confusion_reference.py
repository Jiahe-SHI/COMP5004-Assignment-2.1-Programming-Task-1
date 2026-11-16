"""Correctness checks for the loop confusion-matrix reference."""

import unittest

import numpy as np

from benchmark.stream_references import loop_confusion_matrix
from numcompute_stream.metrics import StreamingConfusionMatrix


class ConfusionReferenceTests(unittest.TestCase):

    def assert_results_equal(self, actual, expected):
        np.testing.assert_array_equal(
            actual["classes"], expected["classes"]
        )
        np.testing.assert_array_equal(
            actual["confusion_matrix"],
            expected["confusion_matrix"],
        )
        self.assertEqual(actual["n_samples"], expected["n_samples"])

    def compare_batch(self, truth, prediction):
        reference = loop_confusion_matrix(truth, prediction)
        metric = StreamingConfusionMatrix()
        metric.update(truth, prediction)
        self.assert_results_equal(metric.result(), reference)
        return reference

    def test_hand_calculated_orientation(self):
        truth = np.array([0, 1, 1, 1])
        prediction = np.array([0, 0, 1, 0])

        reference = self.compare_batch(truth, prediction)

        np.testing.assert_array_equal(reference["classes"], [0, 1])
        np.testing.assert_array_equal(reference["confusion_matrix"], [[1, 0], [2, 1]])
        self.assertEqual(reference["n_samples"], 4)

    def test_strings_and_prediction_only_class(self):
        reference = self.compare_batch(
            np.array(["cat", "cat", "bird"]),
            np.array(["dog", "cat", "cat"]),
        )

        np.testing.assert_array_equal(reference["classes"], ["bird", "cat", "dog"])
        np.testing.assert_array_equal(
            reference["confusion_matrix"],
            [[0, 1, 0], [0, 1, 1], [0, 0, 0]],
        )

    def test_large_integer_labels_remain_distinct(self):
        large = 2**63 + 7
        truth = np.array([-1, large, large + 1], dtype=object)
        prediction = np.array([large, large + 1, -1], dtype=object)

        reference = self.compare_batch(truth, prediction)

        self.assertEqual(reference["classes"].tolist(), [-1, large, large + 1])
        np.testing.assert_array_equal(
            reference["confusion_matrix"],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        )

    def test_seeded_random_multiclass(self):
        rng = np.random.default_rng(2026)

        for n_samples in (1, 17, 257):
            for n_classes in (2, 5):
                with self.subTest(n_samples=n_samples, n_classes=n_classes):
                    truth = rng.integers(0, n_classes, size=n_samples)
                    prediction = rng.integers(0, n_classes, size=n_samples)
                    truth_before = truth.copy()
                    prediction_before = prediction.copy()

                    self.compare_batch(truth, prediction)

                    np.testing.assert_array_equal(truth, truth_before)
                    np.testing.assert_array_equal(prediction, prediction_before)

    def test_unequal_chunks_match_batch_counts(self):
        truth = np.array([5, 5, 2, 1, 5, 2, 1])
        prediction = np.array([5, 2, 2, 5, 1, 1, 1])
        reference = loop_confusion_matrix(truth, prediction)
        metric = StreamingConfusionMatrix()

        for start, stop in ((0, 2), (2, 3), (3, 7)):
            metric.update(truth[start:stop], prediction[start:stop])

        result = metric.result()

        # Streaming classes retain discovery order. Align by labels
        # before comparing with the batch reference's sorted order.
        self.assertEqual(result["classes"].tolist(), [2, 5, 1])
        positions = {
            label: index
            for index, label in enumerate(result["classes"])
        }
        order = [positions[label] for label in reference["classes"]]
        aligned = {
            "classes": result["classes"][order],
            "confusion_matrix": result["confusion_matrix"][
                np.ix_(order, order)
            ],
            "n_samples": result["n_samples"],
        }
        self.assert_results_equal(aligned, reference)


if __name__ == "__main__":
    unittest.main()