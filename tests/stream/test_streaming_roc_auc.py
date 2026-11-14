"""Tests for cumulative binary AUC history and state ownership."""

import unittest
import numpy as np
from numcompute_stream.metrics import StreamingROCAUC, binary_roc_auc


class StreamingROCAUCTests(unittest.TestCase):

    def test_cumulative_auc_is_not_mean_chunk_auc(self):
        metric = StreamingROCAUC(positive_label=1)

        self.assertIs(metric.update([0, 1], [0.1, 0.2]), metric)
        self.assertEqual(metric.result()["auc"], 1.0)

        metric.update([0, 1], [0.8, 0.9])
        result = metric.result()

        # Each chunk has AUC 1.0, but the first positive loses to the
        # second chunk's negative: cumulative AUC is 3 / 4.
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertEqual(result["n_samples"], 4)
        self.assertEqual(result["n_positive"], 2)
        self.assertEqual(result["n_negative"], 2)

    def test_unequal_chunks_match_batch_with_cross_chunk_ties(self):
        labels = np.array([0, 1, 0, 1, 1, 0, 1])
        scores = np.array([0.2, 0.2, 0.6, 0.9, 0.6, 0.1, 0.2])
        metric = StreamingROCAUC(positive_label=1)

        for start, end in ((0, 1), (1, 5), (5, 7)):
            metric.update(labels[start:end], scores[start:end])

        self.assertAlmostEqual(
            metric.result()["auc"],
            binary_roc_auc(labels, scores, positive_label=1),
        )
        self.assertEqual(metric.n_samples_seen_, 7)

    def test_single_class_chunks_and_initial_nan(self):
        metric = StreamingROCAUC(positive_label="yes")
        initial = metric.result()
        self.assertTrue(np.isnan(initial["auc"]))
        self.assertEqual(initial["n_samples"], 0)

        metric.update(["no", "no"], [0.1, 0.4])
        self.assertTrue(np.isnan(metric.result()["auc"]))

        metric.update(["yes"], [0.3])
        self.assertAlmostEqual(metric.result()["auc"], 0.5)

        positive_only = StreamingROCAUC(positive_label=1)
        positive_only.update([1], [0.9])
        self.assertTrue(np.isnan(positive_only.result()["auc"]))

    def test_invalid_updates_preserve_history_and_counts(self):
        metric = StreamingROCAUC(positive_label=1)
        metric.update([0, 1], [0.1, 0.9])
        before = metric.result()

        cases = [
            ([], [], ValueError),
            ([0, 1], [0.1], ValueError),
            ([0], [[0.1]], ValueError),
            ([0], [np.nan], ValueError),
            ([0], [np.inf], ValueError),
            ([0], ["low"], TypeError),
            ([0], [True], TypeError),
            ([None], [0.1], TypeError),
            (["no"], [0.1], TypeError),
            ([2], [0.1], ValueError),
            ([0, 2], [0.1, 0.2], ValueError),
        ]
        for labels, scores, error in cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(error):
                    metric.update(labels, scores)
                self.assertEqual(metric.result(), before)

        metric.update([1], [0.05])
        self.assertAlmostEqual(metric.result()["auc"], 0.5)
        self.assertEqual(metric.n_samples_seen_, 3)

    def test_input_copies_and_query_does_not_learn(self):
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        metric = StreamingROCAUC(positive_label=1)
        metric.update(labels, scores)

        labels[:] = 1
        scores[:] = [0.9, 0.1]

        first = metric.result()
        self.assertEqual(first["auc"], 1.0)
        first["auc"] = -99
        first["n_samples"] = -99

        second = metric.result()
        self.assertEqual(second["auc"], 1.0)
        self.assertEqual(second["n_samples"], 2)
        self.assertEqual(metric.n_samples_seen_, 2)

    def test_reset_retains_positive_class_and_releases_negative(self):
        metric = StreamingROCAUC(positive_label="yes")
        metric.update(["no", "yes"], [0.1, 0.9])

        self.assertIs(metric.reset(), metric)
        self.assertEqual(metric.positive_label, "yes")
        self.assertEqual(metric.n_samples_seen_, 0)
        self.assertTrue(np.isnan(metric.result()["auc"]))

        # A different negative class is allowed after reset.
        metric.update(["other", "yes"], [0.2, 0.8])
        self.assertEqual(metric.result()["auc"], 1.0)
        self.assertEqual(metric.result()["n_samples"], 2)

    def test_large_labels_and_score_precision_guard(self):
        large = 2**63 + 7
        metric = StreamingROCAUC(positive_label=large)
        metric.update(
            np.array([-1, large], dtype=object),
            np.array([0, 1], dtype=np.int64),
        )
        before = metric.result()
        self.assertEqual(before["auc"], 1.0)

        with self.assertRaises(ValueError):
            metric.update(
                np.array([large], dtype=np.uint64),
                np.array([2**63 + 1], dtype=np.uint64),
            )
        self.assertEqual(metric.result(), before)

        # Exactly representable large integers remain allowed.
        metric.update(
            np.array([large], dtype=np.uint64),
            np.array([2**63], dtype=np.uint64),
        )
        self.assertEqual(metric.result()["auc"], 1.0)
        self.assertEqual(metric.n_samples_seen_, 3)

    def test_configuration_and_package_exports(self):
        for positive in (True, 1.0, None):
            with self.subTest(positive=positive):
                with self.assertRaises(TypeError):
                    StreamingROCAUC(positive_label=positive)

        from numcompute_stream import (
            StreamingROCAUC as ExportedAUC,
            binary_roc_auc as exported_function,
        )

        self.assertIs(ExportedAUC, StreamingROCAUC)
        self.assertIs(exported_function, binary_roc_auc)


if __name__ == "__main__":
    unittest.main()