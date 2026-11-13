"""Observable state for successful tree training events."""

import unittest
from unittest.mock import patch
import numpy as np
from numcompute_stream import DecisionTreeClassifier


class TestTreeState(unittest.TestCase):
    def assert_state(self, model, version, retrains, buffered, fitted_rows=None):
        self.assertEqual(model.model_version_, version)
        self.assertEqual(model.n_retrains_, retrains)
        self.assertEqual(model.n_samples_buffered_, buffered)

        if fitted_rows is None:
            self.assertFalse(model.is_fitted_)
            self.assertFalse(hasattr(model, "n_samples_fit_"))
        else:
            self.assertTrue(model.is_fitted_)
            self.assertEqual(model.n_samples_fit_, fitted_rows)

    def test_initial_counters_are_zero_and_readonly(self):
        model = DecisionTreeClassifier()
        self.assert_state(model, 0, 0, 0)

        for name in ("model_version_", "n_retrains_"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    setattr(model, name, 99)

    def test_direct_fit_advances_version_without_resetting_lifetime_count(self):
        model = DecisionTreeClassifier(buffer_size=2)
        model.fit([[0]], [0])
        self.assert_state(model, 1, 0, 0, 1)

        model.partial_fit([[1], [2]], [1, 1])
        self.assert_state(model, 2, 1, 0, 2)

        model.partial_fit([[3]], [1])
        model.fit([[4, 5]], ["new"])
        self.assert_state(model, 3, 1, 0, 1)

    def test_each_whole_buffer_training_counts_once(self):
        model = DecisionTreeClassifier(buffer_size=5)

        for size, expected in ((2, 2), (2, 4)):
            model.partial_fit(np.zeros((size, 1)), np.zeros(size, dtype=int))
            self.assert_state(model, 0, 0, expected)

        model.partial_fit(np.zeros((2, 1)), np.zeros(2, dtype=int))
        self.assert_state(model, 1, 1, 0, 6)

        model.partial_fit(np.zeros((13, 1)), np.ones(13, dtype=int))
        self.assert_state(model, 2, 2, 0, 13)

    def test_prediction_and_ineligible_retry_do_not_change_counters(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.retry_pending_fit()
        self.assert_state(model, 0, 0, 0)

        model.fit([[0], [1]], [0, 1])
        model.partial_fit([[2]], [2])

        for _ in range(2):
            model.predict([[0]])
            model.predict_proba([[0]])
            model.retry_pending_fit()
            self.assert_state(model, 1, 0, 1, 2)

    def test_validation_and_direct_build_failures_do_not_advance_counters(self):
        model = DecisionTreeClassifier(buffer_size=3)
        model.fit([[0]], [0])
        model.partial_fit([[1]], [1])

        with self.assertRaises(ValueError):
            model.fit([[np.nan]], [9])
        self.assert_state(model, 1, 0, 1, 1)

        with self.assertRaises(ValueError):
            model.partial_fit([[np.nan]], [9])
        self.assert_state(model, 1, 0, 1, 1)

        with patch.object(model, "_build_tree", side_effect=RuntimeError("forced")):
            with self.assertRaisesRegex(RuntimeError, "forced"):
                model.fit([[2]], [2])

        self.assert_state(model, 1, 0, 1, 1)

    def test_failed_buffered_training_and_retries_count_only_on_success(self):
        for fitted in (False, True):
            with self.subTest(fitted=fitted):
                model = DecisionTreeClassifier(buffer_size=2)
                if fitted:
                    model.fit([[0]], [0])

                version = int(fitted)
                rows = 1 if fitted else None

                with patch.object(model, "_build_tree", side_effect=RuntimeError("forced"),):
                    with self.assertRaisesRegex(RuntimeError, "forced"):
                        model.partial_fit([[1], [2]], [1, 1])
                    self.assert_state(model, version, 0, 2, rows)

                    with self.assertRaisesRegex(RuntimeError, "forced"):
                        model.retry_pending_fit()
                    self.assert_state(model, version, 0, 2, rows)

                model.retry_pending_fit()
                self.assert_state(model, version + 1, 1, 0, 2)

                model.retry_pending_fit()
                self.assert_state(model, version + 1, 1, 0, 2)


if __name__ == "__main__":
    unittest.main()