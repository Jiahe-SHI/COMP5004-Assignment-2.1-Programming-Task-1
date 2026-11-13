"""Model lifecycle checks for buffered tree training."""

import unittest
from unittest.mock import patch
import numpy as np
from numcompute_stream import DecisionTreeClassifier


class TestTreeLifecycle(unittest.TestCase):
    def test_both_prediction_methods_reject_warmup_without_consuming_buffer(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.partial_fit([[0], [1]], [0, 1])

        for method in (model.predict, model.predict_proba):
            for X in ([[0]], np.empty((0, 1))):
                with self.subTest(method=method.__name__, shape=np.shape(X)):
                    with self.assertRaisesRegex(RuntimeError, "not fitted"):
                        method(X)
                    self.assertFalse(model.is_fitted_)
                    self.assertEqual(model.n_samples_buffered_, 2)

        model.partial_fit([[2]], [1])
        self.assertEqual(model.n_samples_fit_, 3)
        self.assertEqual(model.n_samples_buffered_, 0)

    def test_pending_new_classes_do_not_change_active_probabilities(self):
        model = DecisionTreeClassifier(buffer_size=4, max_depth=0)
        model.fit([[0], [1], [2], [3]], ["a", "a", "a", "b"])
        old_tree = model.tree_
        model.partial_fit([[4], [5]], ["c", "d"])

        for _ in range(2):
            self.assertIs(model.tree_, old_tree)
            np.testing.assert_array_equal(model.classes_, ["a", "b"])
            np.testing.assert_array_equal(model.predict([[0]]), ["a"])
            np.testing.assert_allclose(model.predict_proba([[0]]), [[0.75, 0.25]])
            self.assertEqual(model.predict(np.empty((0, 1))).shape, (0,))
            self.assertEqual(model.predict_proba(np.empty((0, 1))).shape, (0, 2))
            self.assertEqual(model.n_samples_buffered_, 2)

        model.partial_fit([[6], [7]], ["d", "e"])
        np.testing.assert_array_equal(model.classes_, ["c", "d", "e"])
        np.testing.assert_allclose(model.predict_proba([[0]]), [[0.25, 0.5, 0.25]])
        np.testing.assert_array_equal(model.predict([[0]]), ["d"])
        self.assertEqual(model.n_samples_buffered_, 0)

    def test_direct_fit_discards_pending_rows_before_next_training(self):
        model = DecisionTreeClassifier(buffer_size=3, max_depth=0)
        model.partial_fit([[0], [1]], [99, 99])
        model.fit([[10, 20]], ["reset"])

        self.assertEqual(model.n_samples_fit_, 1)
        self.assertEqual(model.n_features_in_, 2)
        self.assertEqual(model.n_samples_buffered_, 0)

        model.partial_fit([[0, 0], [1, 1]], ["new", "new"])
        self.assertEqual(model.n_samples_buffered_, 2)
        np.testing.assert_array_equal(model.predict([[0, 0]]), ["reset"])

        model.partial_fit([[2, 2]], ["new"])
        self.assertEqual(model.n_samples_fit_, 3)
        np.testing.assert_array_equal(model.classes_, ["new"])
        self.assertEqual(model.n_samples_buffered_, 0)

    def test_invalid_direct_fit_preserves_pending_rows_and_probabilities(self):
        model = DecisionTreeClassifier(buffer_size=3, max_depth=0)
        model.fit([[0], [1]], [0, 1])
        model.partial_fit([[2], [3]], [2, 2])
        old_tree = model.tree_

        for X, y in (([[np.nan]], [9]), ([[0]], [])):
            with self.subTest(X=X, y=y):
                with self.assertRaises(ValueError):
                    model.fit(X, y)

                self.assertIs(model.tree_, old_tree)
                self.assertEqual(model.n_samples_buffered_, 2)
                np.testing.assert_allclose(model.predict_proba([[0]]), [[0.5, 0.5]])

        model.partial_fit([[4]], [3])
        self.assertEqual(model.n_samples_fit_, 3)
        np.testing.assert_array_equal(model.classes_, [2, 3])
        np.testing.assert_allclose(model.predict_proba([[0]]), [[2 / 3, 1 / 3]])

    def test_failed_direct_construction_preserves_warmup_or_active_model(self):
        for fitted in (False, True):
            with self.subTest(fitted=fitted):
                model = DecisionTreeClassifier(buffer_size=3, max_depth=0)
                if fitted:
                    model.fit([[0]], [0])

                model.partial_fit([[1], [2]], [1, 1])
                old_tree = getattr(model, "tree_", None)

                with patch.object(model, "_build_tree", side_effect=RuntimeError("forced"),):
                    with self.assertRaisesRegex(RuntimeError, "forced"):
                        model.fit([[8, 9]], ["replacement"])

                self.assertEqual(model.is_fitted_, fitted)
                self.assertEqual(model.n_samples_buffered_, 2)

                if fitted:
                    self.assertIs(model.tree_, old_tree)
                    np.testing.assert_allclose(model.predict_proba([[0]]), [[1.0]])
                else:
                    with self.assertRaisesRegex(RuntimeError, "not fitted"):
                        model.predict_proba([[0]])

                model.partial_fit([[3]], [2])
                self.assertEqual(model.n_samples_fit_, 3)
                np.testing.assert_array_equal(model.classes_, [1, 2])
                np.testing.assert_allclose(model.predict_proba([[0]]), [[2 / 3, 1 / 3]])
                self.assertEqual(model.n_samples_buffered_, 0)


if __name__ == "__main__":
    unittest.main()