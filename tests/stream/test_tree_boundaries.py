"""Combined boundary and consistency checks for the batch tree API."""

import unittest
import numpy as np
from numcompute_stream import DecisionTreeClassifier
from numcompute_stream.tree import _threshold_mask


class TestTreeBoundaries(unittest.TestCase):
    def test_threshold_comparison_matches_python_numeric_order(self):
        thresholds = [
            -(2**64), -(2**63), -257, -1, 0, 255, 256,
            2**53 + 1, 2**63 - 1, 2**64 - 1, 2**64,
            -1e100, -0.5, np.nextafter(0.0, -1.0),
            np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0),
            1.5, 1e100,
        ]
        dtypes = (
            np.int8, np.uint8, np.int32, np.uint32,
            np.int64, np.uint64, np.float16, np.float32, np.float64,
        )
        for dtype in dtypes:
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                values = np.array(
                    [info.min, info.min + 1, 0, info.max - 1, info.max],
                    dtype=dtype,
                )
            else:
                info = np.finfo(dtype)
                values = np.array(
                    [-info.max, -1, -info.smallest_subnormal,
                     -0.0, 0.0, info.smallest_subnormal, 1, info.max],
                    dtype=dtype,
                )

            for threshold in thresholds:
                with self.subTest(dtype=dtype, threshold=threshold):
                    expected = [value.item() <= threshold for value in values]
                    with np.errstate(all="raise"):
                        actual = _threshold_mask(values, threshold)
                    np.testing.assert_array_equal(actual, expected)

    def test_training_row_permutation_preserves_predictions(self):
        X = np.array([
            [0, 3], [0, 3], [1, 2], [1, 2],
            [2, 1], [2, 1], [3, 0], [3, 0],
        ])
        y = np.array(["a", "b", "b", "b", "c", "c", "a", "c"])
        order = np.random.default_rng(8).permutation(len(y))
        query = np.array([
            [0, 3], [1, 2], [2, 1], [3, 0], [1.5, 1.5]
        ])
        for criterion in ("gini", "entropy"):
            with self.subTest(criterion=criterion):
                first = DecisionTreeClassifier(criterion=criterion).fit(X, y)
                second = DecisionTreeClassifier(criterion=criterion).fit(X[order], y[order])

                np.testing.assert_array_equal(first.predict(query), second.predict(query))
                np.testing.assert_allclose(first.predict_proba(query), second.predict_proba(query),)

    def test_chunked_prediction_equals_whole_batch(self):
        X = np.arange(12)[:, None]
        y = np.repeat(["a", "b", "c"], 4)
        model = DecisionTreeClassifier().fit(X, y)
        query = np.array([[11], [0], [7], [4], [3], [8], [1]])
        chunks = (query[:2], query[2:2], query[2:5], query[5:])

        np.testing.assert_array_equal(
            np.concatenate([model.predict(chunk) for chunk in chunks]),
            model.predict(query),
        )
        np.testing.assert_allclose(
            np.concatenate([model.predict_proba(chunk) for chunk in chunks]),
            model.predict_proba(query),
        )

    def test_identical_features_keep_conflicting_label_frequencies(self):
        X = np.array([[-0.0], [0.0], [-0.0]])
        for criterion in ("gini", "entropy"):
            with self.subTest(criterion=criterion):
                model = DecisionTreeClassifier(criterion=criterion).fit(X, ["a", "a", "b"])

                self.assertTrue(model.tree_.is_leaf)
                np.testing.assert_array_equal(model.predict(X), ["a"] * 3)
                np.testing.assert_allclose(
                    model.predict_proba(X),np.tile([2 / 3, 1 / 3], (3, 1)),)

    def test_readonly_noncontiguous_training_and_prediction(self):
        backing = np.array([
            [0, 99, 0, 99], [1, 99, 1, 99],
            [2, 99, 2, 99], [3, 99, 3, 99],
        ])
        backing.setflags(write=False)
        X = backing[:, ::2]
        labels = np.array([10, 99, 10, 99, 20, 99, 20, 99])
        labels.setflags(write=False)
        y = labels[::2]

        self.assertFalse(X.flags.c_contiguous)
        self.assertFalse(y.flags.c_contiguous)

        model = DecisionTreeClassifier().fit(X, y)
        np.testing.assert_array_equal(model.predict(X[::-1]), y[::-1])
        np.testing.assert_array_equal(backing[:, 1::2], np.full((4, 2), 99))

    def test_invalid_refits_leave_public_predictions_usable(self):
        model = DecisionTreeClassifier().fit([[0], [1]], ["a", "b"])
        query = [[1], [0]]
        expected_labels = model.predict(query)
        expected_probabilities = model.predict_proba(query)
        invalid_inputs = (
            ([[np.nan]], ["a"]),
            ([[np.inf]], ["a"]),
            (np.empty((0, 1)), []),
            ([[0], [1]], ["a"]),
            ([[0], [1]], [1, "a"]),
        )

        for X, y in invalid_inputs:
            with self.subTest(X=X, y=y):
                with self.assertRaises((TypeError, ValueError)):
                    model.fit(X, y)
                np.testing.assert_array_equal(model.predict(query), expected_labels)
                np.testing.assert_array_equal(model.predict_proba(query), expected_probabilities,)


if __name__ == "__main__":
    unittest.main()