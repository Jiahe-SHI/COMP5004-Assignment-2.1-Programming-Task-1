"""Tests for vectorised decision tree split search."""

import math
import unittest

import numpy as np

from numcompute_stream.tree import _best_split


def reference_split(X, y, n_classes, criterion):
    """Independent exhaustive oracle for small test inputs."""
    def impurity(labels):
        if len(labels) == 0:
            return 0.0
        counts = [0] * n_classes
        for label in labels:
            counts[int(label)] += 1
        probabilities = [count / len(labels) for count in counts if count]
        if criterion == "gini":
            return 1.0 - sum(p * p for p in probabilities)
        return -sum(p * math.log2(p) for p in probabilities)

    parent = impurity(y)
    candidates = []
    for feature in range(X.shape[1]):
        for boundary in sorted(set(X[:, feature].tolist()))[:-1]:
            mask = X[:, feature] <= boundary
            left = y[mask]
            right = y[~mask]
            gain = (
                parent - len(left) / len(y) * impurity(left) - len(right) / len(y) * impurity(right)
            )
            candidates.append((gain, feature, mask))

    maximum = max((item[0] for item in candidates), default=0.0)
    if maximum <= 1e-12:
        return None
    for gain, feature, mask in candidates:
        if gain > 1e-12 and gain >= maximum - 1e-12:
            return gain, feature, mask
    return None


class TestSplitSearch(unittest.TestCase):
    def test_perfect_binary_split(self):
        X = np.array([[0], [1], [2], [3]])
        for criterion, expected in (("gini", 0.5), ("entropy", 1.0)):
            with self.subTest(criterion=criterion):
                result = _best_split(X, [0, 0, 1, 1], 2, criterion=criterion)
                self.assertEqual(result["feature"], 0)
                self.assertAlmostEqual(result["gain"], expected)
                self.assertEqual(result["n_left"], 2)
                self.assertEqual(result["n_right"], 2)
                np.testing.assert_array_equal(
                    X[:, 0] <= result["threshold"], [True, True, False, False]
                )

    def test_duplicate_values_are_not_separated(self):
        X = np.array([[0], [0], [1], [1]])
        result = _best_split(X, [0, 0, 1, 1], 2)
        self.assertEqual(result["n_left"], 2)
        np.testing.assert_array_equal(
            X[:, 0] <= result["threshold"], [True, True, False, False]
        )

    def test_constant_pure_and_single_sample_return_none(self):
        cases = (
            ([[1], [1]], [0, 1]),
            ([[0], [1]], [0, 0]),
            ([[1]], [0]),
        )
        for X, y in cases:
            with self.subTest(X=X, y=y):
                self.assertIsNone(_best_split(X, y, 2))

    def test_zero_gain_returns_none(self):
        self.assertIsNone(
            _best_split([[0], [0], [1], [1]], [0, 1, 0, 1], 2)
        )

    def test_feature_ties_use_smallest_index(self):
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        result = _best_split(X, [0, 0, 1, 1], 2, feature_indices=[1, 0])
        self.assertEqual(result["feature"], 0)

    def test_position_ties_use_first_boundary(self):
        result = _best_split([[0], [1], [2]], [0, 1, 0], 2)
        self.assertEqual(result["n_left"], 1)
        self.assertAlmostEqual(result["gain"], 1 / 9)

    def test_feature_subset_is_respected(self):
        X = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])
        self.assertIsNone(
            _best_split(X, [0, 0, 1, 1], 2, feature_indices=[1])
        )

    def test_multiclass_split(self):
        result = _best_split(
            np.arange(6).reshape(-1, 1), [0, 0, 1, 1, 2, 2], 3
        )
        self.assertEqual(result["n_left"], 2)
        self.assertAlmostEqual(result["gain"], 1 / 3)

    def test_extreme_integer_thresholds_preserve_partition(self):
        arrays = (
            np.array([-(2**63), 2**63 - 1], dtype=np.int64),
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            np.array([2**64 - 2, 2**64 - 1], dtype=np.uint64),
        )
        for values in arrays:
            with self.subTest(values=values):
                with np.errstate(all="raise"):
                    result = _best_split(values[:, None], [0, 1], 2)
                    mask = values <= result["threshold"]
                self.assertIsInstance(result["threshold"], int)
                np.testing.assert_array_equal(mask, [True, False])

    def test_extreme_float_thresholds_preserve_partition(self):
        arrays = (
            np.array([-np.finfo(float).max, np.finfo(float).max]),
            np.array([1e308, 1.6e308]),
            np.array([1.0, np.nextafter(1.0, 2.0)]),
            np.array([0.0, np.nextafter(0.0, 1.0)]),
            np.array(
                [np.float32(1), np.nextafter(np.float32(1), np.float32(2))],
                dtype=np.float32,
            ),
        )
        for values in arrays:
            with self.subTest(values=values):
                with np.errstate(all="raise"):
                    result = _best_split(values[:, None], [0, 1], 2)
                    mask = values <= result["threshold"]
                self.assertTrue(np.isfinite(result["threshold"]))
                np.testing.assert_array_equal(mask, [True, False])

    def test_matches_independent_exhaustive_reference(self):
        rng = np.random.default_rng(42)
        for criterion in ("gini", "entropy"):
            for trial in range(20):
                X = rng.integers(-3, 4, size=(12, 3))
                y = rng.integers(0, 3, size=12)
                with self.subTest(criterion=criterion, trial=trial):
                    expected = reference_split(X, y, 3, criterion)
                    actual = _best_split(X, y, 3, criterion=criterion)
                    if expected is None:
                        self.assertIsNone(actual)
                    else:
                        gain, feature, mask = expected
                        self.assertIsNotNone(actual)
                        self.assertEqual(actual["feature"], feature)
                        self.assertAlmostEqual(actual["gain"], gain)
                        actual_mask = X[:, feature] <= actual["threshold"]
                        np.testing.assert_array_equal(actual_mask, mask)
                        self.assertEqual(actual["n_left"], int(mask.sum()))

    def test_rejects_invalid_inputs(self):
        for codes in ([0], [[0], [1]], [-1, 1], [0, 2]):
            with self.subTest(codes=codes):
                with self.assertRaises(ValueError):
                    _best_split([[0], [1]], codes, 2)
        with self.assertRaises(TypeError):
            _best_split([[0], [1]], [0.0, 1.0], 2)
        with self.assertRaises(ValueError):
            _best_split([[np.nan], [1]], [0, 1], 2)
        with self.assertRaises(ValueError):
            _best_split([[0], [1]], [0, 1], 2, criterion="unknown")
        for features in ([0, 0], [-1], [1]):
            with self.subTest(features=features):
                with self.assertRaises(ValueError):
                    _best_split(
                        [[0], [1]], [0, 1], 2, feature_indices=features
                    )

    def test_does_not_modify_inputs(self):
        X = np.array([[2], [0], [1]])
        y = np.array([1, 0, 0])
        X.setflags(write=False)
        y.setflags(write=False)
        _best_split(X, y, 2)
        np.testing.assert_array_equal(X[:, 0], [2, 0, 1])
        np.testing.assert_array_equal(y, [1, 0, 0])


if __name__ == "__main__":
    unittest.main()