"""Tests for array-storage accounting and memory log fields."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import DecisionTreeClassifier
from numcompute_stream._memory import array_storage_bytes
from numcompute_stream.stream import StreamTrainer


class TestStreamTrainerMemory(unittest.TestCase):
    def test_shared_references_and_cycles_are_counted_once(self):
        array = np.zeros(10, dtype=np.float64)
        root = {"first": array, "again": [array]}
        root["cycle"] = root

        self.assertEqual(array_storage_bytes(root), 80)

    def test_distinct_views_use_logical_array_sizes(self):
        base = np.arange(10, dtype=np.int64)
        first = base[:6]
        second = base[4:]

        # Overlapping views are distinct arrays under this accounting method.
        self.assertEqual(
            array_storage_bytes([first, second]),
            first.nbytes + second.nbytes,
        )

    def test_object_referents_are_excluded(self):
        referenced = np.zeros(1000, dtype=np.float64)
        array = np.empty(1, dtype=object)
        array[0] = referenced

        self.assertEqual(array_storage_bytes(array), array.nbytes)

    def test_pending_buffer_bytes_clear_after_training(self):
        trainer = StreamTrainer(
            DecisionTreeClassifier(buffer_size=3)
        )
        X = np.array([[0.0], [1.0]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.int64)

        trainer.fit_chunk(X, y)
        pending = trainer.memory_result()

        self.assertEqual(pending["buffer_array_bytes"], X.nbytes + y.nbytes)
        self.assertGreaterEqual(pending["memory_bytes"], pending["buffer_array_bytes"])

        trainer.fit_chunk([[2.0]], [1])
        trained = trainer.memory_result()

        self.assertEqual(trained["buffer_array_bytes"], 0)
        self.assertGreater(trained["memory_bytes"], 0)
        self.assertEqual(trained["log_events"], 2)

    def test_logs_include_measurement_scope_and_prior_event_count(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
        record = trainer.score_chunk([[0.0]], [0])
        trainer.fit_chunk([[0.0]], [0])

        self.assertEqual(record["memory_method"], "reachable_ndarray_nbytes")
        self.assertEqual(record["memory_status"], "ok")
        self.assertEqual(record["memory_log_events_before"], 0)

        logs = trainer.logs_
        self.assertEqual(logs[1]["memory_log_events_before"], 1)
        self.assertGreater(logs[1]["buffer_array_bytes"], 0)

        snapshot = trainer.memory_result()
        snapshot["memory_bytes"] = -1
        self.assertGreaterEqual(trainer.memory_result()["memory_bytes"], 0)

    def test_cumulative_auc_history_increases_retained_array_bytes(self):
        model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
        trainer = StreamTrainer(model, positive_label=1, window_size=2)
        X = np.array([[0.0], [1.0]])
        y = np.array([0, 1])

        trainer.score_chunk(X, y)
        first = trainer.memory_result()

        trainer.score_chunk(X, y)
        second = trainer.memory_result()

        self.assertGreater(second["memory_bytes"], first["memory_bytes"])
        self.assertEqual(second["buffer_array_bytes"], 0)
        self.assertEqual(trainer.metrics_result()["rolling_auc"]["n_samples"], 2)
        self.assertEqual(trainer.metrics_result()["cumulative_auc"]["n_samples"], 4)

    def test_memory_failure_does_not_change_training_or_mask_errors(self):
        trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
        with patch(
            "numcompute_stream.stream.array_storage_bytes",
            side_effect=RuntimeError("measurement failed"),
        ):
            trainer.fit_chunk([[0.0], [1.0]], [0, 1])

        self.assertTrue(trainer.model.is_fitted_)
        successful = trainer.logs_[-1]
        self.assertTrue(successful["success"])
        self.assertEqual(successful["memory_status"], "unavailable")
        self.assertIsNone(successful["memory_bytes"])
        self.assertEqual(successful["memory_error_type"], "RuntimeError")

        original_error = ValueError("original learning error")
        with patch(
            "numcompute_stream.stream.array_storage_bytes",
            side_effect=RuntimeError("measurement failed"),
        ):
            with patch.object(
                DecisionTreeClassifier,
                "partial_fit",
                side_effect=original_error,
            ):
                with self.assertRaises(ValueError) as caught:
                    trainer.fit_chunk([[2.0]], [1])

        self.assertIs(caught.exception, original_error)
        failed = trainer.logs_[-1]
        self.assertFalse(failed["success"])
        self.assertEqual(failed["error_type"], "ValueError")
        self.assertEqual(failed["memory_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()