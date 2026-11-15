"""Tests for separate streaming evaluation and learning."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    OneHotEncoder,
    Pipeline,
)
from numcompute_stream.metrics import RollingClassificationMetrics
from numcompute_stream.stream import StreamTrainer


class TestStreamTrainer(unittest.TestCase):
    def test_warmup_is_excluded_from_evaluation_denominator(self):
        model = DecisionTreeClassifier(buffer_size=2)
        trainer = StreamTrainer(model, window_size=3)

        first = trainer.score_chunk([[0.0]], [0])
        self.assertTrue(first["warm_up"])
        self.assertTrue(np.isnan(first["chunk_accuracy"]))
        self.assertTrue(np.isnan(first["cumulative_accuracy"]))
        self.assertEqual(trainer.n_samples_evaluated_, 0)
        self.assertEqual(model.n_samples_buffered_, 0)

        trainer.fit_chunk([[0.0]], [0])
        second = trainer.score_chunk([[1.0]], [1])
        self.assertTrue(second["warm_up"])
        trainer.fit_chunk([[1.0]], [1])

        third = trainer.score_chunk([[0.0]], [0])
        self.assertFalse(third["warm_up"])
        self.assertEqual(third["chunk_accuracy"], 1.0)
        self.assertEqual(trainer.n_chunks_scored_, 3)
        self.assertEqual(trainer.n_samples_scored_, 3)
        self.assertEqual(trainer.n_samples_evaluated_, 1)

    def test_scoring_uses_model_before_current_chunk_learning(self):
        model = DecisionTreeClassifier(buffer_size=2)
        model.fit([[0.0], [1.0]], [0, 0])
        trainer = StreamTrainer(model)

        record = trainer.score_chunk([[10.0], [11.0]], [1, 1])

        self.assertEqual(record["chunk_accuracy"], 0.0)
        self.assertEqual(record["model_version"], 1)
        self.assertEqual(model.model_version_, 1)
        np.testing.assert_array_equal(model.classes_, [0])

        trainer.fit_chunk([[10.0], [11.0]], [1, 1])

        np.testing.assert_array_equal(model.classes_, [1])
        self.assertEqual(model.model_version_, 2)
        self.assertEqual(trainer.n_samples_evaluated_, 2)
        self.assertEqual(trainer.metrics_result()["cumulative"]["accuracy"], 0.0)

    def test_fit_chunk_never_updates_evaluation_metrics(self):
        model = DecisionTreeClassifier(buffer_size=2)
        trainer = StreamTrainer(model)

        self.assertIs(trainer.fit_chunk([[0.0], [1.0]], [0, 1]), trainer)
        self.assertTrue(model.is_fitted_)
        self.assertIs(trainer.model, model)
        self.assertEqual(trainer.n_chunks_scored_, 0)
        self.assertEqual(trainer.n_samples_scored_, 0)
        self.assertEqual(trainer.n_samples_evaluated_, 0)
        self.assertTrue(np.isnan(trainer.metrics_result()["cumulative"]["accuracy"]))

    def test_unequal_chunks_use_sample_counts_and_rolling_window(self):
        model = DecisionTreeClassifier().fit([[0.0]], [0])
        trainer = StreamTrainer(model, window_size=2)

        trainer.score_chunk([[0.0]], [0])
        record = trainer.score_chunk([[1.0], [2.0], [3.0]], [1, 1, 0])

        self.assertAlmostEqual(record["chunk_accuracy"], 1 / 3)
        self.assertAlmostEqual(record["cumulative_accuracy"], 2 / 4)
        self.assertAlmostEqual(record["rolling_accuracy"], 1 / 2)

        result = trainer.metrics_result()
        self.assertEqual(result["cumulative"]["n_samples"], 4)
        self.assertEqual(result["rolling"]["n_samples"], 2)

    def test_metric_failure_rolls_back_both_accumulators(self):
        model = DecisionTreeClassifier().fit([[0.0]], [0])
        trainer = StreamTrainer(model)
        trainer.score_chunk([[0.0]], [0])

        with patch.object(
            RollingClassificationMetrics,
            "update",
            side_effect=RuntimeError("simulated rolling failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                trainer.score_chunk([[1.0], [2.0]], [1, 1])

        self.assertEqual(trainer.n_chunks_scored_, 1)
        self.assertEqual(trainer.n_samples_scored_, 1)
        self.assertEqual(trainer.n_samples_evaluated_, 1)
        self.assertEqual(trainer.metrics_result()["cumulative"]["accuracy"], 1.0)

        trainer.score_chunk([[1.0], [2.0]], [1, 1])
        self.assertEqual(trainer.n_chunks_scored_, 2)
        self.assertEqual(trainer.n_samples_evaluated_, 3)
        self.assertAlmostEqual(trainer.metrics_result()["cumulative"]["accuracy"], 1 / 3)

    def test_categorical_pipeline_supports_new_evaluation_labels(self):
        model = Pipeline([
            ("encoder", OneHotEncoder()),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        trainer = StreamTrainer(model)

        warmup = trainer.score_chunk([["a"], ["b"]], [0, 1])
        self.assertTrue(warmup["warm_up"])
        trainer.fit_chunk([["a"], ["b"]], [0, 1])

        record = trainer.score_chunk([["c"]], [2])

        self.assertEqual(record["chunk_accuracy"], 0.0)
        self.assertEqual(trainer.n_samples_evaluated_, 1)
        self.assertEqual(model.n_samples_buffered_, 0)
        np.testing.assert_array_equal(model.classes_, [0, 1])
        self.assertIn(2, trainer.metrics_result()["cumulative"]["classes"].tolist())

    def test_invalid_scoring_does_not_advance_state(self):
        model = DecisionTreeClassifier()
        trainer = StreamTrainer(model)

        for X, y in (
            ([[0.0]], [0, 1]),
            ([[np.inf]], [0]),
            (np.empty((0, 1)), np.array([], dtype=int)),
            ([[0.0]], [True]),
        ):
            with self.subTest(X=X, y=y):
                with self.assertRaises((TypeError, ValueError)):
                    trainer.score_chunk(X, y)
                self.assertEqual(trainer.n_chunks_scored_, 0)
                self.assertEqual(trainer.n_samples_scored_, 0)
                self.assertEqual(model.n_samples_buffered_, 0)

    def test_results_are_independent_and_repeat_scoring_is_explicit(self):
        model = DecisionTreeClassifier().fit([[0.0]], [0])
        trainer = StreamTrainer(model)

        record = trainer.score_chunk([[0.0]], [0])
        record["n_samples_evaluated"] = 999
        result = trainer.metrics_result()
        result["cumulative"]["confusion_matrix"][:] = 999
        result["rolling"]["classes"][:] = 999

        self.assertEqual(trainer.n_samples_evaluated_, 1)
        np.testing.assert_array_equal(
            trainer.metrics_result()["cumulative"]["confusion_matrix"],
            [[1]],
        )
        np.testing.assert_array_equal(trainer.metrics_result()["rolling"]["classes"], [0])

        # A second explicit scoring call is a second evaluation event.
        trainer.score_chunk([[0.0]], [0])
        self.assertEqual(trainer.n_samples_evaluated_, 2)
        self.assertEqual(trainer.n_chunks_scored_, 2)
        self.assertEqual(model.model_version_, 1)


if __name__ == "__main__":
    unittest.main()