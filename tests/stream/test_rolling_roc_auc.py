"""Tests for binary ROC AUC over sample-based rolling windows."""

import unittest

import numpy as np

from numcompute_stream.metrics import RollingROCAUC, binary_roc_auc


class RollingROCAUCTests(unittest.TestCase):

    def test_partial_eviction_changes_auc(self):
        metric = RollingROCAUC(3, positive_label=1)
        self.assertIs(metric.update([0, 1, 0], [0.1, 0.3, 0.4]), metric,)
        self.assertAlmostEqual(metric.result()["auc"], 0.5)

        metric.update([1, 0], [0.8, 0.9])
        result = metric.result()

        # Retained labels: [0, 1, 0]; scores: [0.4, 0.8, 0.9].
        self.assertAlmostEqual(result["auc"], 0.5)
        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["n_positive"], 1)
        self.assertEqual(result["n_negative"], 2)
        self.assertEqual(result["n_samples_seen"], 5)

        metric.update([1], [1.0])
        self.assertEqual(metric.result()["auc"], 0.5)

        metric.update([0], [0.2])
        self.assertEqual(metric.result()["auc"], 1.0)

    def test_oversized_chunk_retains_tail_and_ties(self):
        metric = RollingROCAUC(3, positive_label=1)
        metric.update([0, 1], [0.1, 0.9])
        metric.update(
            [0, 1, 0, 1, 0],
            [0.0, 1.0, 0.5, 0.5, 0.2],
        )
        result = metric.result()

        # Positive 0.5 ties negative 0.5 and beats negative 0.2.
        self.assertAlmostEqual(result["auc"], 0.75)
        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["n_samples_seen"], 7)

        # Structural checks ensure evicted samples are not retained.
        self.assertEqual(len(metric._label_chunks), 1)
        self.assertEqual(len(metric._score_chunks), 1)
        self.assertEqual(metric._label_chunks[0].size, 3)
        self.assertEqual(metric._score_chunks[0].size, 3)
        self.assertIsNone(metric._label_chunks[0].base)
        self.assertIsNone(metric._score_chunks[0].base)

    def test_single_class_window_becomes_nan_and_recovers(self):
        metric = RollingROCAUC(2, positive_label=1)
        self.assertTrue(np.isnan(metric.result()["auc"]))

        metric.update([0, 1], [0.1, 0.9])
        self.assertEqual(metric.result()["auc"], 1.0)

        metric.update([1], [0.8])
        result = metric.result()
        self.assertTrue(np.isnan(result["auc"]))
        self.assertEqual(result["n_positive"], 2)
        self.assertEqual(result["n_negative"], 0)

        metric.update([0], [0.2])
        self.assertEqual(metric.result()["auc"], 1.0)

        single = RollingROCAUC(1, positive_label=1)
        single.update([0, 1], [0.1, 0.9])
        self.assertTrue(np.isnan(single.result()["auc"]))
        self.assertEqual(single.result()["n_samples"], 1)
        self.assertEqual(single.n_samples_seen_, 2)

    def test_unequal_chunks_match_latest_window_reference(self):
        labels = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.3, 0.4, 0.8, 0.4, 0.2, 0.9, 0.9, 0.5])
        metric = RollingROCAUC(4, positive_label=1)

        for start, end in ((0, 2), (2, 5), (5, 6), (6, 9)):
            metric.update(labels[start:end], scores[start:end])
            expected = binary_roc_auc(
                labels[max(0, end - 4):end],
                scores[max(0, end - 4):end],
                positive_label=1,
            )
            self.assertAlmostEqual(metric.result()["auc"], expected)
            self.assertEqual(metric.n_samples_seen_, end)
            self.assertEqual(metric.result()["n_samples"], min(4, end))

    def test_invalid_full_chunk_preserves_state(self):
        metric = RollingROCAUC(2, positive_label=1)
        metric.update([0, 1], [0.1, 0.9])
        before = metric.result()

        cases = [
            ([], [], ValueError),
            ([0, 1], [0.1], ValueError),
            ([None, 0, 1], [0.2, 0.3, 0.9], TypeError),
            ([0, 0, 1], [np.nan, 0.3, 0.9], ValueError),
            ([2, 0, 1], [0.2, 0.3, 0.9], ValueError),
            ([0], [True], TypeError),
            (["no"], [0.1], TypeError),
            ([0], np.array([2**63 + 1], dtype=np.uint64), ValueError),
        ]
        for labels, scores, error in cases:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(error):
                    metric.update(labels, scores)
                self.assertEqual(metric.result(), before)

    def test_negative_label_persists_until_reset(self):
        metric = RollingROCAUC(2, positive_label="yes")
        metric.update(["no"], [0.1])
        metric.update(["yes", "yes"], [0.8, 0.9])

        # The original negative is outside the window, but still defines
        # the binary task until reset.
        with self.assertRaises(ValueError):
            metric.update(["other"], [0.2])
        self.assertEqual(metric.n_samples_seen_, 3)
        self.assertTrue(np.isnan(metric.result()["auc"]))

        self.assertIs(metric.reset(), metric)
        self.assertEqual(metric.window_size, 2)
        self.assertEqual(metric.positive_label, "yes")
        self.assertEqual(metric.n_samples_seen_, 0)
        self.assertTrue(np.isnan(metric.result()["auc"]))

        metric.update(["other", "yes"], [0.2, 0.8])
        self.assertEqual(metric.result()["auc"], 1.0)

    def test_input_ownership_and_queries_do_not_update(self):
        labels = np.array([0, 1])
        scores = np.array([0.1, 0.9])
        metric = RollingROCAUC(3, positive_label=1)
        metric.update(labels, scores)

        labels[:] = 1
        scores[:] = [0.9, 0.1]
        first = metric.result()
        self.assertEqual(first["auc"], 1.0)
        first["auc"] = -99
        first["n_samples_seen"] = -99

        self.assertEqual(metric.result()["auc"], 1.0)
        self.assertEqual(metric.n_samples_seen_, 2)

        metric.update([0], [0.5])
        self.assertEqual(metric.result()["auc"], 1.0)
        self.assertEqual(metric.result()["n_samples"], 3)

    def test_configuration_and_package_export(self):
        for size in (True, 2.5, "3"):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    RollingROCAUC(size, positive_label=1)
        for size in (0, -1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    RollingROCAUC(size, positive_label=1)

        with self.assertRaises(TypeError):
            RollingROCAUC(3, positive_label=True)

        from numcompute_stream import RollingROCAUC as ExportedAUC
        self.assertIs(ExportedAUC, RollingROCAUC)


if __name__ == "__main__":
    unittest.main()