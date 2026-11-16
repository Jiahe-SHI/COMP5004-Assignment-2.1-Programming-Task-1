"""Compare loop and vectorised split evaluation."""

import unittest
import numpy as np
from benchmark.stream_references import loop_best_split
from numcompute_stream.tree import _best_split


class SplitReferenceTests(unittest.TestCase):

    def compare(self, X, y, n_classes=2, criterion="gini"):
        X = np.asarray(X)
        y = np.asarray(y)
        expected = loop_best_split(X, y, n_classes, criterion=criterion)
        actual = _best_split(X, y, n_classes, criterion=criterion)

        if expected is None:
            self.assertIsNone(actual)
            return None

        self.assertIsNotNone(actual)
        self.assertEqual(actual["feature"], expected["feature"])
        self.assertEqual(actual["n_left"], expected["n_left"])
        self.assertEqual(actual["n_right"], expected["n_right"])
        self.assertAlmostEqual(actual["gain"], expected["gain"], delta=1e-12)

        feature = expected["feature"]
        np.testing.assert_array_equal(
            X[:, feature] <= actual["threshold"],
            X[:, feature] <= expected["boundary"],
        )
        return expected

    def test_hand_calculated_perfect_split(self):
        for criterion, gain in (("gini", 0.5), ("entropy", 1.0)):
            with self.subTest(criterion=criterion):
                result = self.compare(
                    [[0], [1], [2], [3]], [0, 0, 1, 1],
                    criterion=criterion,
                )
                self.assertEqual(result["feature"], 0)
                self.assertEqual(result["n_left"], 2)
                self.assertAlmostEqual(result["gain"], gain)

    def test_no_positive_split(self):
        cases = (
            ([[1], [1]], [0, 1]),
            ([[0], [1]], [0, 0]),
            ([[1]], [0]),
            ([[0], [0], [1], [1]], [0, 1, 0, 1]),
        )
        for criterion in ("gini", "entropy"):
            for X, y in cases:
                with self.subTest(criterion=criterion, X=X, y=y):
                    self.assertIsNone(
                        self.compare(X, y, criterion=criterion)
                    )

    def test_duplicate_values_stay_together(self):
        for criterion in ("gini", "entropy"):
            with self.subTest(criterion=criterion):
                result = self.compare(
                    [[0], [0], [1], [1]], [0, 0, 1, 1],
                    criterion=criterion,
                )
                self.assertEqual(result["n_left"], 2)
                self.assertEqual(result["n_right"], 2)

    def test_feature_and_position_ties(self):
        for criterion in ("gini", "entropy"):
            with self.subTest(criterion=criterion):
                result = self.compare(
                    [[0, 0], [1, 1], [2, 2]], [0, 1, 0],
                    criterion=criterion,
                )
                self.assertEqual(result["feature"], 0)
                self.assertEqual(result["n_left"], 1)

    def test_seeded_random_multiclass(self):
        rng = np.random.default_rng(2026)
        for criterion in ("gini", "entropy"):
            for discrete in (False, True):
                for trial in range(10):
                    with self.subTest(
                        criterion=criterion,
                        discrete=discrete,
                        trial=trial,
                    ):
                        if discrete:
                            X = rng.integers(-3, 4, size=(31, 4))
                        else:
                            X = rng.normal(size=(31, 4))
                        y = rng.integers(0, 3, size=31)
                        self.compare(X, y, 3, criterion)

    def test_readonly_inputs_are_unchanged(self):
        X = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        y = np.array([1, 0, 0])
        original_X = X.copy()
        original_y = y.copy()
        X.setflags(write=False)
        y.setflags(write=False)

        self.compare(X, y)

        np.testing.assert_array_equal(X, original_X)
        np.testing.assert_array_equal(y, original_y)


if __name__ == "__main__":
    unittest.main()