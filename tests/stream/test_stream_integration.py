"""End-to-end streaming coordination and recovery tests."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
    StreamTrainer,
)
from numcompute_stream.stream import StreamTrainer as ModuleStreamTrainer


def pairwise_auc(y, scores, positive_label=1):
    """Small independent reference; not a production implementation."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    positive = scores[y == positive_label]
    negative = scores[y != positive_label]
    if positive.size == 0 or negative.size == 0:
        return float("nan")

    credit = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                credit += 1.0
            elif positive_score == negative_score:
                credit += 0.5
    return credit / (positive.size * negative.size)


def numeric_trainer():
    return StreamTrainer(
        Pipeline([
            ("imputer", Imputer()),
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier(
                buffer_size=4, random_state=7
            )),
        ]),
        window_size=3,
        positive_label=1,
    )


class TestStreamIntegration(unittest.TestCase):
    def test_unequal_numeric_chunks_match_recorded_prediction_reference(self):
        self.assertIs(StreamTrainer, ModuleStreamTrainer)
        trainer = numeric_trainer()
        chunks = [
            ([[0.0], [1.0]], [0, 0]),
            ([[2.0], [3.0], [4.0]], [0, 1, 1]),
            ([[np.nan]], [1]),
            ([[0.0], [4.0], [2.0], [5.0]], [0, 1, 0, 1]),
            ([[1.0]], [0]),
        ]
        truths = []
        predictions = []
        positive_scores = []
        records = []

        for X, y in chunks:
            model = trainer.model
            version_before = model.model_version_

            # Capture predictions from the model available BEFORE learning.
            if model.is_fitted_:
                predicted = model.predict(X)
                probabilities = model.predict_proba(X)
                classes = model.classes_.tolist()
                if 1 in classes:
                    scores = probabilities[:, classes.index(1)]
                else:
                    scores = np.zeros(len(y))

                truths.extend(y)
                predictions.extend(predicted.tolist())
                positive_scores.extend(scores.tolist())

            record = trainer.score_chunk(X, y)
            self.assertEqual(record["model_version"], version_before)
            self.assertEqual(model.model_version_, version_before)
            records.append(record)
            trainer.fit_chunk(X, y)

        self.assertEqual(
            [record["warm_up"] for record in records],
            [True, True, False, False, False],
        )
        self.assertEqual(
            [record["model_version"] for record in records],
            [0, 0, 1, 1, 2],
        )
        self.assertEqual(trainer.n_samples_scored_, 11)
        self.assertEqual(trainer.n_samples_evaluated_, 6)
        self.assertEqual(trainer.model.n_retrains_, 2)
        self.assertEqual(trainer.model.n_samples_fit_, 5)

        # The final one-row buffer is retained without automatic flushing.
        self.assertEqual(trainer.model.n_samples_buffered_, 1)

        truth = np.asarray(truths)
        prediction = np.asarray(predictions)
        scores = np.asarray(positive_scores)
        result = trainer.metrics_result()

        self.assertAlmostEqual(
            result["cumulative"]["accuracy"],
            float(np.mean(truth == prediction)),
        )
        self.assertAlmostEqual(
            result["rolling"]["accuracy"],
            float(np.mean(truth[-3:] == prediction[-3:])),
        )
        self.assertAlmostEqual(
            result["cumulative_auc"]["auc"],
            pairwise_auc(truth, scores),
        )
        self.assertAlmostEqual(
            result["rolling_auc"]["auc"],
            pairwise_auc(truth[-3:], scores[-3:]),
        )

        logs = trainer.logs_
        self.assertEqual(
            [row["event"] for row in logs],
            ["score", "fit"] * 5,
        )
        self.assertEqual(
            [row["retrained"] for row in logs if row["event"] == "fit"],
            [False, True, False, True, False],
        )
        for row in logs:
            self.assertEqual(row["memory_status"], "ok")
            self.assertEqual(row["memory_method"], "reachable_ndarray_nbytes")
            self.assertGreaterEqual(row["memory_bytes"], row["buffer_array_bytes"])

    def test_changing_categories_preserve_prelearning_evaluation(self):
        trainer = StreamTrainer(
            Pipeline([
                ("encoder", OneHotEncoder()),
                ("tree", DecisionTreeClassifier(buffer_size=2)),
            ]),
            window_size=2,
        )
        chunks = [
            ([["a"], ["b"]], [0, 1]),
            ([["a"], ["c"]], [0, 2]),
            ([["c"], ["d"]], [2, 3]),
        ]
        expected_truth = []
        expected_prediction = []

        for X, y in chunks:
            if trainer.model.is_fitted_:
                expected_truth.extend(y)
                expected_prediction.extend(
                    trainer.model.predict(X).tolist()
                )

            previous_version = trainer.model.model_version_
            record = trainer.score_chunk(X, y)
            self.assertEqual(record["model_version"], previous_version)
            trainer.fit_chunk(X, y)

        self.assertEqual(trainer.n_samples_evaluated_, 4)
        self.assertEqual(trainer.model.n_retrains_, 3)
        self.assertEqual(trainer.model.n_samples_buffered_, 0)
        np.testing.assert_array_equal(trainer.model.classes_, [2, 3])

        result = trainer.metrics_result()
        np.testing.assert_array_equal(result["cumulative"]["classes"], [0,1,2,3])
        self.assertAlmostEqual(
            result["cumulative"]["accuracy"],
            float(np.mean(
                np.asarray(expected_truth) == np.asarray(expected_prediction)
            )),
        )
        self.assertIsNone(result["cumulative_auc"])

    def test_training_retry_does_not_repeat_successful_scoring(self):
        model = Pipeline([
            ("imputer", Imputer()),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        trainer = StreamTrainer(model, positive_label=1)

        trainer.score_chunk([[0.0], [1.0]], [0, 1])
        trainer.fit_chunk([[0.0], [1.0]], [0, 1])

        # Opposite labels make the pre-learning predictions incorrect.
        record = trainer.score_chunk([[0.0], [1.0]], [1, 0])
        self.assertEqual(record["chunk_accuracy"], 0.0)
        self.assertEqual(record["cumulative_auc"], 0.0)

        with patch.object(
            DecisionTreeClassifier,
            "fit",
            side_effect=RuntimeError("simulated retraining failure"),
        ):
            with self.assertRaises(RuntimeError):
                trainer.fit_chunk([[0.0], [1.0]], [1, 0])

        self.assertEqual(model.n_samples_buffered_, 2)
        self.assertEqual(model.model_version_, 1)
        self.assertEqual(trainer.n_samples_evaluated_, 2)

        # Only retry learning. The scoring event already succeeded.
        trainer.retry_pending_fit()

        self.assertEqual(model.n_samples_buffered_, 0)
        self.assertEqual(model.n_samples_fit_, 2)
        self.assertEqual(model.model_version_, 2)
        self.assertEqual(trainer.n_chunks_scored_, 2)
        self.assertEqual(trainer.n_samples_evaluated_, 2)

        result = trainer.metrics_result()
        self.assertEqual(result["cumulative"]["accuracy"], 0.0)
        self.assertEqual(result["cumulative_auc"]["auc"], 0.0)
        self.assertEqual(result["cumulative_auc"]["n_samples"], 2)

        np.testing.assert_array_equal(model.predict([[0.0], [1.0]]), [1, 0])
        logs = trainer.logs_
        self.assertEqual(
            [row["event"] for row in logs],
            ["score", "fit", "score", "fit", "retry_fit"],
        )
        self.assertFalse(logs[3]["success"])
        self.assertTrue(logs[4]["success"])
        self.assertTrue(logs[4]["retrained"])

    def test_independent_trainers_do_not_share_learning_or_metrics(self):
        first = numeric_trainer()
        second = numeric_trainer()
        X = [[0.0], [1.0], [2.0], [3.0]]
        y = [0, 0, 1, 1]

        for trainer in (first, second):
            trainer.score_chunk(X, y)
            trainer.fit_chunk(X, y)
            trainer.score_chunk(X, y)

        second_predictions = second.model.predict(X)
        second_version = second.model.model_version_
        second_log_count = len(second.logs_)

        first.fit_chunk(X, [1, 1, 0, 0])
        first.score_chunk(X, y)

        self.assertEqual(second.model.model_version_, second_version)
        self.assertEqual(len(second.logs_), second_log_count)
        self.assertEqual(second.n_samples_evaluated_, 4)
        self.assertEqual(first.n_samples_evaluated_, 8)
        np.testing.assert_array_equal(second.model.predict(X), second_predictions)
        self.assertEqual(second.metrics_result()["cumulative"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()