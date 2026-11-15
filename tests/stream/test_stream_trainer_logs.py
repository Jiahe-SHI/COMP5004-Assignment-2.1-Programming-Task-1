"""Tests for trainer event logs, timing scope and retry behaviour."""

import unittest
from unittest.mock import patch

from numcompute_stream import DecisionTreeClassifier, Pipeline
from numcompute_stream.stream import StreamTrainer


class TestStreamTrainerLogs(unittest.TestCase):
    def test_score_then_fit_records_order_and_model_versions(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))

        trainer.score_chunk([[0.0], [1.0]], [0, 1])
        trainer.fit_chunk([[0.0], [1.0]], [0, 1])

        logs = trainer.logs_
        self.assertEqual([row["event"] for row in logs], ["score", "fit"])
        self.assertEqual([row["event_id"] for row in logs], [1, 2])
        self.assertTrue(logs[0]["warm_up"])
        self.assertEqual(logs[0]["model_version"], 0)
        self.assertEqual(logs[0]["n_samples_evaluated"], 0)
        self.assertEqual(logs[1]["model_version_before"], 0)
        self.assertEqual(logs[1]["model_version"], 1)
        self.assertTrue(logs[1]["retrained"])
        self.assertEqual(logs[1]["buffer_samples"], 0)

    def test_score_timing_uses_elapsed_clock_difference(self):
        trainer = StreamTrainer(DecisionTreeClassifier())

        with patch(
            "numcompute_stream.stream.perf_counter",
            side_effect=[10.0, 10.25],
        ):
            record = trainer.score_chunk([[0.0]], [0])

        self.assertEqual(record["score_seconds"], 0.25)
        self.assertEqual(trainer.logs_[0]["score_seconds"], 0.25)

    def test_fit_timing_and_buffer_only_event(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=3))

        with patch(
            "numcompute_stream.stream.perf_counter",
            side_effect=[20.0, 20.5],
        ):
            trainer.fit_chunk([[0.0]], [0])

        record = trainer.logs_[0]
        self.assertEqual(record["training_seconds"], 0.5)
        self.assertFalse(record["retrained"])
        self.assertEqual(record["buffer_samples_before"], 0)
        self.assertEqual(record["buffer_samples"], 1)
        self.assertEqual(record["model_version"], 0)
        self.assertEqual(trainer.n_samples_evaluated_, 0)

    def test_returned_records_and_log_lists_are_independent(self):
        trainer = StreamTrainer(DecisionTreeClassifier())

        record = trainer.score_chunk([[0.0]], [0])
        record["warm_up"] = False
        copied_logs = trainer.logs_
        copied_logs[0]["event"] = "changed"
        copied_logs.clear()

        self.assertEqual(len(trainer.logs_), 1)
        self.assertEqual(trainer.logs_[0]["event"], "score")
        self.assertTrue(trainer.logs_[0]["warm_up"])

    def test_failed_score_keeps_metrics_and_logs_unchanged(self):
        model = DecisionTreeClassifier().fit([[0.0]], [0])
        trainer = StreamTrainer(model)
        trainer.score_chunk([[0.0]], [0])

        with patch.object(
            DecisionTreeClassifier,
            "predict",
            side_effect=RuntimeError("simulated prediction failure"),
        ):
            with self.assertRaises(RuntimeError):
                trainer.score_chunk([[1.0]], [1])

        self.assertEqual(len(trainer.logs_), 1)
        self.assertEqual(trainer.n_chunks_scored_, 1)
        self.assertEqual(trainer.n_samples_evaluated_, 1)

    def test_failed_training_and_retry_are_separate_events(self):
        model = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2)),])
        trainer = StreamTrainer(model)
        original_error = RuntimeError("simulated training failure")

        with patch.object(DecisionTreeClassifier, "fit", side_effect=original_error):
            with self.assertRaises(RuntimeError) as caught:
                trainer.fit_chunk([[0.0], [1.0]], [0, 1])

        self.assertIs(caught.exception, original_error)
        failed = trainer.logs_[0]
        self.assertFalse(failed["success"])
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertEqual(failed["buffer_samples"], 2)
        self.assertFalse(failed["retrained"])

        self.assertIs(trainer.retry_pending_fit(), trainer)

        retried = trainer.logs_[1]
        self.assertEqual(retried["event"], "retry_fit")
        self.assertTrue(retried["success"])
        self.assertTrue(retried["retrained"])
        self.assertEqual(retried["buffer_samples_before"], 2)
        self.assertEqual(retried["buffer_samples"], 0)
        self.assertEqual(model.n_samples_fit_, 2)
        self.assertEqual(trainer.n_samples_evaluated_, 0)

    def test_invalid_training_is_logged_without_buffer_acceptance(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=3))
        trainer.fit_chunk([[0.0]], [0])

        with self.assertRaises(ValueError):
            trainer.fit_chunk([[1.0]], [0, 1])

        record = trainer.logs_[-1]
        self.assertFalse(record["success"])
        self.assertEqual(record["error_type"], "ValueError")
        self.assertEqual(record["buffer_samples_before"], 1)
        self.assertEqual(record["buffer_samples"], 1)
        self.assertFalse(record["retrained"])

    def test_ineligible_retry_is_logged_without_retraining(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=3))
        trainer.fit_chunk([[0.0]], [0])
        trainer.retry_pending_fit()

        record = trainer.logs_[-1]
        self.assertEqual(record["event"], "retry_fit")
        self.assertTrue(record["success"])
        self.assertFalse(record["retrained"])
        self.assertEqual(record["buffer_samples"], 1)
        self.assertEqual(record["model_version"], 0)


if __name__ == "__main__":
    unittest.main()