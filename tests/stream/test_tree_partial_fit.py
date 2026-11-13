"""Tests for whole-buffer retraining."""

import unittest
from unittest.mock import patch
import numpy as np
from numcompute_stream import DecisionTreeClassifier


class TestTreePartialFit(unittest.TestCase):
    def test_threshold_overshoot_uses_all_600_rows(self):
        model = DecisionTreeClassifier(buffer_size=500)
        X = np.arange(200).reshape(-1, 1)

        with patch.object(model, "fit", wraps=model.fit) as fit:
            for expected in (200, 400):
                self.assertIs(model.partial_fit(X, np.zeros(200, dtype=int)), model)
                self.assertEqual(model.n_samples_buffered_, expected)
                self.assertFalse(model.is_fitted_)

            fit.assert_not_called()
            model.partial_fit(X, np.ones(200, dtype=int))
            fit.assert_called_once()

        self.assertEqual(model.n_samples_fit_, 600)
        self.assertEqual(model.n_samples_buffered_, 0)
        np.testing.assert_allclose(model.predict_proba([[0]]), [[2 / 3, 1 / 3]])

    def test_exact_threshold_and_large_chunk_each_train_once(self):
        for size in (5, 13):
            with self.subTest(size=size):
                model = DecisionTreeClassifier(buffer_size=5)
                with patch.object(model, "fit", wraps=model.fit) as fit:
                    model.partial_fit(np.zeros((size, 1)), np.zeros(size, dtype=int))
                    fit.assert_called_once()

                self.assertEqual(model.n_samples_fit_, size)
                self.assertEqual(model.n_samples_buffered_, 0)

    def test_multiple_rounds_replace_model_without_history(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.partial_fit([[0], [1], [2]], [0, 0, 0])
        old_tree = model.tree_

        model.partial_fit([[3]], [1])
        self.assertIs(model.tree_, old_tree)
        np.testing.assert_array_equal(model.predict([[0]]), [0])

        model.partial_fit([[4], [5]], [1, 1])
        self.assertIsNot(model.tree_, old_tree)
        self.assertEqual(model.n_samples_fit_, 3)
        np.testing.assert_array_equal(model.classes_, [1])
        self.assertEqual(model.n_samples_buffered_, 0)

    def test_short_final_buffer_remains_unfitted(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.partial_fit([[0], [1]], [0, 1])

        self.assertEqual(model.n_samples_buffered_, 2)
        with self.assertRaises(RuntimeError):
            model.predict([[0]])

    def test_mixed_signed_unsigned_labels_remain_exact(self):
        model = DecisionTreeClassifier(buffer_size=2)
        model.partial_fit([[0]], np.array([-1], dtype=np.int64))
        model.partial_fit([[1]], np.array([2**64 - 1], dtype=np.uint64))

        self.assertEqual(model.classes_.tolist(), [-1, 2**64 - 1])
        self.assertEqual(model.predict([[0], [1]]).tolist(), [-1, 2**64 - 1])

    def test_different_string_lengths_are_not_truncated(self):
        model = DecisionTreeClassifier(buffer_size=2)
        model.partial_fit([[0]], np.array(["a"]))
        model.partial_fit([[1]], np.array(["long_label"]))

        self.assertEqual(model.predict([[0], [1]]).tolist(), ["a", "long_label"])

    def test_numeric_feature_promotion_and_precision_guard(self):
        model = DecisionTreeClassifier(buffer_size=2)
        model.partial_fit(np.array([[1]], dtype=np.int32), [0])
        model.partial_fit(np.array([[2.5]], dtype=np.float64), [1])
        self.assertEqual(model.predict([[1], [2.5]]).tolist(), [0, 1])

        model = DecisionTreeClassifier(buffer_size=2)
        model.partial_fit(
            np.array([[2**53 + 1]], dtype=np.int64), [0]
        )
        with self.assertRaisesRegex(ValueError, "precision"):
            model.partial_fit(np.array([[0.5]], dtype=np.float64), [1])

        self.assertFalse(model.is_fitted_)
        self.assertEqual(model.n_samples_buffered_, 2)

    def test_successful_fit_clears_pending_data_and_schema(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.partial_fit([[0]], [0])
        model.fit([[0, 1]], ["new"])

        self.assertEqual(model.n_samples_buffered_, 0)
        model.partial_fit([[2, 3]], ["next"])
        self.assertEqual(model.n_samples_buffered_, 1)

    def test_failed_training_keeps_old_tree_and_entire_buffer(self):
        model = DecisionTreeClassifier(buffer_size=2).fit([[0]], [0])
        old_tree = model.tree_
        model.partial_fit([[1]], [1])

        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.partial_fit([[2]], [1])

        self.assertIs(model.tree_, old_tree)
        self.assertEqual(model.n_samples_buffered_, 2)
        self.assertEqual(model.predict([[0]]).tolist(), [0])


if __name__ == "__main__":
    unittest.main()