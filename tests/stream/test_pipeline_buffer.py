"""Behaviour tests for numeric Pipeline buffering."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    StandardScaler,
)
from numcompute_stream.pipeline import Pipeline


def make_pipeline(buffer_size=5):
    return Pipeline([
        ("imputer", Imputer()),
        ("scaler", StandardScaler()),
        ("tree", DecisionTreeClassifier(
            buffer_size=buffer_size, random_state=7
        )),
    ])


class TestPipelineBuffer(unittest.TestCase):
    def test_warmup_and_short_tail_do_not_train(self):
        pipeline = make_pipeline()
        self.assertFalse(pipeline.is_fitted_)
        self.assertEqual(pipeline.model_version_, 0)

        self.assertIs(pipeline.partial_fit([[0.0], [1.0]], [0, 1]), pipeline)
        self.assertEqual(pipeline.n_samples_buffered_, 2)
        self.assertIs(pipeline.retry_pending_fit(), pipeline)
        self.assertFalse(pipeline.is_fitted_)

        with self.assertRaises(RuntimeError):
            pipeline.predict([[0.0]])
        self.assertEqual(pipeline.n_samples_buffered_, 2)

    def test_overshoot_uses_entire_buffer_with_one_fit(self):
        pipeline = make_pipeline(buffer_size=5)
        with patch.object(
            DecisionTreeClassifier,
            "partial_fit",
            side_effect=AssertionError("Tree must not buffer again"),
        ):
            pipeline.partial_fit([[0.0], [1.0]], [0, 0])
            pipeline.partial_fit([[2.0], [3.0]], [0, 1])
            pipeline.partial_fit([[4.0], [5.0]], [1, 1])

        self.assertEqual(pipeline.n_samples_fit_, 6)
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.n_retrains_, 1)
        self.assertEqual(pipeline.model_version_, 1)
        np.testing.assert_array_equal(
            pipeline.predict(np.arange(6.0).reshape(-1, 1)),
            [0, 0, 0, 1, 1, 1],
        )

    def test_exact_threshold_and_oversized_chunk(self):
        for size in (5, 8):
            with self.subTest(size=size):
                pipeline = make_pipeline(buffer_size=5)
                pipeline.partial_fit(
                    np.arange(size).reshape(-1, 1),
                    np.zeros(size, dtype=int),
                )
                self.assertEqual(pipeline.n_samples_fit_, size)
                self.assertEqual(pipeline.n_retrains_, 1)
                self.assertEqual(pipeline.n_samples_buffered_, 0)

    def test_pending_chunks_are_independent_copies(self):
        pipeline = make_pipeline(buffer_size=3)
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])
        pipeline.partial_fit(X, y)

        X[:] = 100.0
        y[:] = 9
        pipeline.partial_fit([[2.0]], [1])

        np.testing.assert_array_equal(pipeline.classes_, [0, 1])
        np.testing.assert_array_equal(pipeline.predict([[0.0], [1.0], [2.0]]), [0, 1, 1])

    def test_old_chain_stays_active_between_training_events(self):
        pipeline = make_pipeline(buffer_size=4)
        pipeline.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        before = pipeline.predict_proba([[0.0], [3.0]])

        pipeline.partial_fit([[1000.0], [2000.0]], [2, 2])

        np.testing.assert_array_equal(pipeline.predict_proba([[0.0], [3.0]]), before)
        np.testing.assert_array_equal(pipeline.classes_, [0, 1])
        self.assertEqual(pipeline.model_version_, 1)

        pipeline.partial_fit([[3000.0], [4000.0]], [2, 2])
        np.testing.assert_array_equal(pipeline.classes_, [2])
        self.assertEqual(pipeline.n_samples_fit_, 4)
        self.assertEqual(pipeline.n_retrains_, 2)

    def test_failed_training_retains_old_model_and_retry_data(self):
        pipeline = make_pipeline(buffer_size=2)
        pipeline.partial_fit([[0.0], [1.0]], [0, 1])
        before = pipeline.predict([[0.0], [1.0]])

        with patch.object(
            DecisionTreeClassifier, "fit",
            side_effect=RuntimeError("simulated training failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                pipeline.partial_fit([[10.0], [11.0]], [2, 2])

        self.assertEqual(pipeline.n_samples_buffered_, 2)
        self.assertEqual(pipeline.model_version_, 1)
        self.assertEqual(pipeline.n_retrains_, 1)
        np.testing.assert_array_equal(pipeline.predict([[0.0], [1.0]]), before)

        pipeline.retry_pending_fit()
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.n_samples_fit_, 2)
        self.assertEqual(pipeline.n_retrains_, 2)
        np.testing.assert_array_equal(pipeline.classes_, [2])

    def test_invalid_chunk_does_not_enter_buffer(self):
        pipeline = make_pipeline()
        pipeline.partial_fit([[0.0]], [0])

        invalid_pairs = [
            ([[1.0, 2.0]], [0]),
            ([[np.inf]], [0]),
            ([[1.0]], ["changed"]),
            ([[1.0]], [0, 1]),
        ]
        for X, y in invalid_pairs:
            with self.subTest(X=X, y=y):
                with self.assertRaises((TypeError, ValueError)):
                    pipeline.partial_fit(X, y)
                self.assertEqual(pipeline.n_samples_buffered_, 1)
                self.assertFalse(pipeline.is_fitted_)

    def test_direct_fit_clears_pending_only_on_success(self):
        pipeline = make_pipeline()
        pipeline.partial_fit([[0.0]], [0])

        with patch.object(
            DecisionTreeClassifier, "fit",
            side_effect=RuntimeError("simulated"),
        ):
            with self.assertRaises(RuntimeError):
                pipeline.fit([[10.0]], [1])
        self.assertEqual(pipeline.n_samples_buffered_, 1)
        self.assertFalse(pipeline.is_fitted_)

        self.assertIs(pipeline.fit([[10.0, 20.0]], ["new"]), pipeline)
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.n_retrains_, 0)
        self.assertEqual(pipeline.model_version_, 1)
        np.testing.assert_array_equal(pipeline.predict([[10.0, 20.0]]), ["new"])


if __name__ == "__main__":
    unittest.main()