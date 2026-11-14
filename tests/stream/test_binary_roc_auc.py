"""Tests for binary ROC AUC and tied-score handling."""

import unittest
import numpy as np
from numcompute_stream.metrics import binary_roc_auc


class BinaryRocAucTests(unittest.TestCase):

    def test_perfect_reversed_and_hand_calculated_scores(self):
        labels = [0, 0, 1, 1]
        self.assertEqual(binary_roc_auc(labels, [0.1, 0.2, 0.8, 0.9], positive_label=1), 1.0,)
        self.assertEqual(binary_roc_auc(labels, [0.9, 0.8, 0.2, 0.1], positive_label=1), 0.0,)
        self.assertAlmostEqual(
            binary_roc_auc(
                [0, 1, 0, 1],
                [0.1, 0.3, 0.35, 0.8],
                positive_label=1,
            ),
            0.75,
        )

    def test_ties_receive_half_credit(self):
        self.assertEqual(binary_roc_auc([0, 1, 0, 1], [5, 5, 5, 5], positive_label=1), 0.5,)
        self.assertAlmostEqual(binary_roc_auc([0, 1, 0, 1], [0, 0, 1, 2], positive_label=1), 0.625,)

    def test_explicit_positive_class_and_unbounded_scores(self):
        labels = ["yes", "no", "yes", "no"]
        scores = [4, -3, 2, -1]
        self.assertEqual(binary_roc_auc(labels, scores, positive_label="yes"), 1.0,)
        self.assertEqual(binary_roc_auc(labels, scores, positive_label="no"), 0.0,)
        self.assertEqual(binary_roc_auc(labels, [-4, 3, -2, 1], positive_label="no"), 1.0,)

    def test_undefined_when_either_class_is_missing(self):
        for labels, scores in (
            ([], []),
            ([1, 1], [0.2, 0.8]),
            ([0, 0], [0.2, 0.8]),
        ):
            with self.subTest(labels=labels):
                self.assertTrue(np.isnan(binary_roc_auc(labels, scores, positive_label=1)))

    def test_matches_independent_pairwise_reference(self):
        rng = np.random.default_rng(42)
        labels = np.array([0, 1] * 15)
        scores = rng.integers(-3, 4, size=labels.size)

        positives = scores[labels == 1]
        negatives = scores[labels == 0]

        # Deliberately simple test oracle; production code does not
        # construct or iterate over all positive-negative pairs.
        credit = sum(
            1.0 if p > n else 0.5 if p == n else 0.0
            for p in positives
            for n in negatives
        )
        expected = credit / (positives.size * negatives.size)
        actual = binary_roc_auc(labels, scores, positive_label=1)
        self.assertAlmostEqual(actual, expected)

        permutation = rng.permutation(labels.size)
        self.assertAlmostEqual(
            binary_roc_auc(
                labels[permutation],
                scores[permutation],
                positive_label=1,
            ),
            expected,
        )

    def test_large_integer_labels_and_adjacent_integer_scores(self):
        large = 2**63 + 7
        labels = np.array([-1, large], dtype=object)
        scores = np.array([2**63, 2**63 + 1], dtype=np.uint64)

        self.assertEqual(binary_roc_auc(labels, scores, positive_label=large), 1.0,)
        self.assertEqual(binary_roc_auc(labels, scores[::-1], positive_label=large), 0.0,)

    def test_readonly_noncontiguous_inputs_are_not_modified(self):
        labels_base = np.array([0, 9, 1, 9, 0, 9, 1, 9])
        scores_base = np.array([0.1, 9, 0.3, 9, 0.35, 9, 0.8, 9])
        labels = labels_base[::2]
        scores = scores_base[::2]
        labels.setflags(write=False)
        scores.setflags(write=False)
        before_labels = labels_base.copy()
        before_scores = scores_base.copy()

        self.assertAlmostEqual(binary_roc_auc(labels, scores, positive_label=1), 0.75,)
        np.testing.assert_array_equal(labels_base, before_labels)
        np.testing.assert_array_equal(scores_base, before_scores)

    def test_invalid_inputs_are_rejected(self):
        cases = [
            ([0, 1], [0.1], 1, ValueError),
            ([0, 1], [[0.1], [0.9]], 1, ValueError),
            ([0, 1], [0.1, np.nan], 1, ValueError),
            ([0, 1], [0.1, np.inf], 1, ValueError),
            ([0, 1], ["low", "high"], 1, TypeError),
            ([0, 1], [False, True], 1, TypeError),
            ([0, 1], [1j, 2j], 1, TypeError),
            ([0.0, 1.0], [0.1, 0.9], 1, TypeError),
            ([0, None], [0.1, 0.9], 1, TypeError),
            ([0, 1], [0.1, 0.9], "yes", TypeError),
            ([0, 1], [0.1, 0.9], True, TypeError),
            ([0, 1, 2], [0.1, 0.5, 0.9], 1, ValueError),
            ([0, 2], [0.1, 0.9], 1, ValueError),
        ]
        for labels, scores, positive, error in cases:
            with self.subTest(labels=labels, scores=scores, positive=positive):
                with self.assertRaises(error):
                    binary_roc_auc(labels, scores, positive_label=positive)


if __name__ == "__main__":
    unittest.main()