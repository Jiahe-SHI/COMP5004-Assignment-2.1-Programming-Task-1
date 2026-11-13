"""Failure recovery and caller-owned input checks."""

import unittest
from unittest.mock import patch
import numpy as np
from numcompute_stream import DecisionTreeClassifier


class TestTreeRecovery(unittest.TestCase):
    def test_retry_initial_failure_without_duplicate_samples(self):
        model = DecisionTreeClassifier(buffer_size=4, max_depth=0)
        model.partial_fit([[0], [1]], [0, 0])

        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit([[2], [3]], [0, 1])

        self.assertFalse(model.is_fitted_)
        self.assertEqual(model.n_samples_buffered_, 4)
        self.assertIs(model.retry_pending_fit(), model)
        self.assertEqual(model.n_samples_fit_, 4)
        self.assertEqual(model.n_samples_buffered_, 0)
        np.testing.assert_allclose(model.predict_proba([[0]]), [[0.75, 0.25]])

        with patch.object(model, "fit", wraps=model.fit) as fit:
            model.retry_pending_fit()
            fit.assert_not_called()

    def test_repeated_failures_keep_old_model_until_success(self):
        model = DecisionTreeClassifier(buffer_size=2, max_depth=0)
        model.fit([[0], [1]], [0, 1])
        old_tree = model.tree_

        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit([[2], [3]], [2, 2])

            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "forced"):
                    model.retry_pending_fit()
                self.assertIs(model.tree_, old_tree)
                self.assertEqual(model.n_samples_buffered_, 2)
                np.testing.assert_allclose(model.predict_proba([[0]]), [[0.5, 0.5]])

        model.retry_pending_fit()
        self.assertEqual(model.n_samples_fit_, 2)
        self.assertEqual(model.n_samples_buffered_, 0)
        np.testing.assert_array_equal(model.classes_, [2])

    def test_retry_does_not_flush_empty_or_short_buffers(self):
        for fitted in (False, True):
            with self.subTest(fitted=fitted):
                model = DecisionTreeClassifier(buffer_size=3)
                if fitted:
                    model.fit([[0]], [0])
                old_tree = getattr(model, "tree_", None)

                with patch.object(model, "fit", wraps=model.fit) as fit:
                    self.assertIs(model.retry_pending_fit(), model)
                    model.partial_fit([[1], [2]], [1, 1])
                    self.assertIs(model.retry_pending_fit(), model)
                    fit.assert_not_called()

                self.assertEqual(model.n_samples_buffered_, 2)
                self.assertIs(getattr(model, "tree_", None), old_tree)

    def test_input_mutation_before_and_after_failure_cannot_change_training(self):
        model = DecisionTreeClassifier(buffer_size=4)
        X_first = np.array([[0], [1]])
        y_first = np.array([0, 0])
        model.partial_fit(X_first, y_first)

        X_first[:] = 100
        y_first[:] = 9

        X_last = np.array([[2], [3]])
        y_last = np.array([1, 1])
        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit(X_last, y_last)

        X_last[:] = -100
        y_last[:] = 8

        model.retry_pending_fit()
        np.testing.assert_array_equal(model.classes_, [0, 1])
        np.testing.assert_array_equal(model.predict([[0], [1], [2], [3]]), [0, 0, 1, 1])
        self.assertEqual(model.n_samples_fit_, 4)
        self.assertEqual(model.n_samples_buffered_, 0)

    def test_invalid_new_chunk_does_not_change_failed_buffer(self):
        model = DecisionTreeClassifier(buffer_size=2, max_depth=0)
        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit([[0], [1]], [0, 1])

        with self.assertRaises(ValueError):
            model.partial_fit([[np.nan]], [9])

        self.assertEqual(model.n_samples_buffered_, 2)
        self.assertFalse(model.is_fitted_)

        model.retry_pending_fit()
        self.assertEqual(model.n_samples_fit_, 2)
        np.testing.assert_allclose(model.predict_proba([[0]]), [[0.5, 0.5]])

    def test_retry_revalidates_configuration_without_losing_pending_data(self):
        model = DecisionTreeClassifier(buffer_size=2)
        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit([[0], [1]], [0, 1])

        model.min_samples_split = 1
        with self.assertRaises(ValueError):
            model.retry_pending_fit()

        self.assertEqual(model.n_samples_buffered_, 2)
        self.assertFalse(model.is_fitted_)

        model.min_samples_split = 2
        model.retry_pending_fit()
        self.assertEqual(model.n_samples_buffered_, 0)
        np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])


if __name__ == "__main__":
    unittest.main()