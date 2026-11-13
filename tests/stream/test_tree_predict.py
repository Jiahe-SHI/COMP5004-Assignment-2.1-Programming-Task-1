"""Tests for batch labels, probabilities and prediction state isolation."""

import pickle
import unittest

import numpy as np

from numcompute_stream import DecisionTreeClassifier
from numcompute_stream.tree import DecisionTreeClassifier as TreeClassifier


class TestTreePrediction(unittest.TestCase):
    def test_package_export(self):
        self.assertIs(DecisionTreeClassifier, TreeClassifier)

    def test_predict_restores_string_labels_and_row_order(self):
        model = DecisionTreeClassifier().fit(
            [[0], [1], [2], [3]], ["low", "low", "high", "high"]
        )
        result = model.predict([[3], [0], [2], [1], [3]])
        np.testing.assert_array_equal(
            result, ["high", "low", "high", "low", "high"]
        )
        self.assertEqual(result.shape, (5,))

    def test_multiclass_probabilities_follow_sorted_classes(self):
        model = DecisionTreeClassifier().fit(
            np.arange(6)[:, None], ["z", "z", "a", "a", "m", "m"]
        )
        np.testing.assert_array_equal(model.classes_, ["a", "m", "z"])
        probabilities = model.predict_proba([[4], [0], [2]])
        np.testing.assert_array_equal(
            probabilities, [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        )
        np.testing.assert_array_equal(
            model.predict([[4], [0], [2]]), ["m", "z", "a"]
        )

    def test_mixed_leaf_returns_frequencies(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0], [1], [2], [3]], ["b", "a", "b", "b"]
        )
        np.testing.assert_allclose(
            model.predict_proba([[100], [-100]]),
            [[0.25, 0.75], [0.25, 0.75]],
        )
        np.testing.assert_array_equal(
            model.predict([[100], [-100]]), ["b", "b"]
        )

    def test_vote_tie_uses_first_class(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0], [1]], [20, 10]
        )
        np.testing.assert_array_equal(model.predict([[9]]), [10])
        np.testing.assert_array_equal(
            model.predict_proba([[9]]), [[0.5, 0.5]]
        )

    def test_single_class_probability_is_one(self):
        model = DecisionTreeClassifier().fit([[0]], ["only"])
        np.testing.assert_array_equal(
            model.predict([[-1], [9]]), ["only", "only"]
        )
        np.testing.assert_array_equal(
            model.predict_proba([[-1], [9]]), [[1], [1]]
        )

    def test_empty_prediction_shapes_and_label_dtype(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        labels = model.predict(np.empty((0, 1)))
        probabilities = model.predict_proba(np.empty((0, 1)))
        self.assertEqual(labels.shape, (0,))
        self.assertEqual(labels.dtype, model.classes_.dtype)
        self.assertEqual(probabilities.shape, (0, 2))
        self.assertEqual(probabilities.dtype, np.dtype("float64"))

    def test_unfitted_prediction_raises_clear_error(self):
        model = DecisionTreeClassifier()
        for method in (model.predict, model.predict_proba):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(RuntimeError, "not fitted"):
                    method([[0]])
                with self.assertRaisesRegex(RuntimeError, "not fitted"):
                    method(np.empty((0, 1)))

    def test_invalid_prediction_inputs_are_rejected(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        for method in (model.predict, model.predict_proba):
            inputs = (
                [0],
                [[0, 1]],
                np.empty((0, 2)),
                [[np.nan]],
                [[np.inf]],
            )
            for X in inputs:
                with self.subTest(method=method.__name__, X=X):
                    with self.assertRaises(ValueError):
                        method(X)
            with self.assertRaises(TypeError):
                method([["0"]])

    def test_prediction_and_output_changes_do_not_modify_model(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [10, 20])
        before = pickle.dumps(model)
        X = np.array([[1], [0]])
        X.setflags(write=False)
        labels = model.predict(X)
        probabilities = model.predict_proba(X)
        labels[:] = -1
        probabilities[:] = -1

        self.assertEqual(pickle.dumps(model), before)
        np.testing.assert_array_equal(X, [[1], [0]])
        np.testing.assert_array_equal(model.predict(X), [20, 10])
        np.testing.assert_array_equal(
            model.predict_proba(X), [[0, 1], [1, 0]]
        )

    def test_large_integer_labels_are_preserved(self):
        y = [2**80, 2**80 + 1]
        model = DecisionTreeClassifier().fit([[0], [1]], y)
        self.assertEqual(
            model.predict([[1], [0]]).tolist(), [y[1], y[0]]
        )

        unsigned = np.array(
            [2**64 - 2, 2**64 - 1], dtype=np.uint64
        )
        model.fit([[0], [1]], unsigned)
        result = model.predict([[1], [0]])
        self.assertEqual(result.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(result, unsigned[::-1])

    def test_threshold_equality_goes_left(self):
        model = DecisionTreeClassifier().fit([[0.0], [2.0]], [0, 1])
        boundary = model.tree_.threshold
        X = np.array([
            [boundary],
            [np.nextafter(boundary, np.inf)],
        ])
        np.testing.assert_array_equal(model.predict(X), [0, 1])

    def test_float64_threshold_with_float32_prediction(self):
        boundary_upper = np.nextafter(1.0, 0.0)
        model = DecisionTreeClassifier().fit(
            [[0.0], [2 * boundary_upper]], [0, 1]
        )
        self.assertLess(model.tree_.threshold, 1.0)
        X = np.array([[1.0], [0.0]], dtype=np.float32)
        np.testing.assert_array_equal(model.predict(X), [1, 0])

    def test_integer_threshold_with_float_prediction(self):
        base = 2**53
        # Create an odd integer threshold not representable in float64.
        X = np.array(
            [[base + 3], [base + 4]], dtype=np.int64
        )
        model = DecisionTreeClassifier().fit(X, [0, 1])
        query = np.array(
            [[base + 4], [base + 2]], dtype=np.float64
        )
        np.testing.assert_array_equal(model.predict(query), [1, 0])

    def test_float_threshold_with_large_integer_prediction(self):
        base = 2**53
        model = DecisionTreeClassifier().fit(
            np.array(
                [[base - 2], [base + 2]], dtype=np.float64
            ),
            [0, 1],
        )
        self.assertEqual(model.tree_.threshold, base)
        query = np.array([[base + 1], [base]], dtype=np.int64)
        np.testing.assert_array_equal(model.predict(query), [1, 0])

    def test_threshold_outside_prediction_dtype_range(self):
        model = DecisionTreeClassifier().fit(
            [[-10], [-8]], [0, 1]
        )
        np.testing.assert_array_equal(
            model.predict(np.array([[0], [255]], dtype=np.uint8)),
            [1, 1],
        )

        model.fit([[1000], [2000]], [0, 1])
        np.testing.assert_array_equal(
            model.predict(np.array([[0], [255]], dtype=np.uint8)),
            [0, 0],
        )

        for sign, expected in ((1, 0), (-1, 1)):
            # Use sorted labels so the left leaf is always class 0.
            training = (
                [[-2e100], [-1e100]]
                if sign < 0
                else [[1e100], [2e100]]
            )
            model.fit(training, [0, 1])
            with np.errstate(all="raise"):
                actual = model.predict(
                    np.array([[0]], dtype=np.float16)
                )
            np.testing.assert_array_equal(actual, [expected])

    def test_matches_scalar_tree_traversal_for_both_criteria(self):
        rng = np.random.default_rng(123)
        X = rng.integers(-4, 5, size=(40, 4))
        y = rng.integers(0, 3, size=40)
        query = rng.normal(size=(25, 4))

        for criterion in ("gini", "entropy"):
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=4,
                random_state=17,
            ).fit(X, y)

            expected_labels = []
            expected_probabilities = []

            # Independent scalar traversal for test comparison only.
            for row in query:
                node = model.tree_
                while not node.is_leaf:
                    if row[node.feature].item() <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                expected_labels.append(
                    model.classes_[node.prediction]
                )
                expected_probabilities.append(
                    node.counts / node.n_samples
                )

            actual = model.predict_proba(query)
            np.testing.assert_array_equal(
                model.predict(query), expected_labels
            )
            np.testing.assert_allclose(
                actual, expected_probabilities
            )
            np.testing.assert_allclose(actual.sum(axis=1), 1.0)
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())
            np.testing.assert_array_equal(
                model.predict(query),
                model.classes_[np.argmax(actual, axis=1)],
            )

    def test_refit_predictions_use_new_classes_and_feature_count(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        model.fit([[0, 0], [1, 1]], ["new", "new"])
        np.testing.assert_array_equal(
            model.predict([[9, 9]]), ["new"]
        )
        self.assertEqual(
            model.predict_proba([[9, 9]]).shape, (1, 1)
        )
        with self.assertRaises(ValueError):
            model.predict([[9]])


if __name__ == "__main__":
    unittest.main()