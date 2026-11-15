"""Trainer-level binary AUC alignment and coordinated failure recovery."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    OneHotEncoder,
    Pipeline,
)
from numcompute_stream.metrics import RollingROCAUC
from numcompute_stream.stream import StreamTrainer


class TestStreamTrainerAUC(unittest.TestCase):
    def test_positive_label_can_be_first_probability_column(self):
        model = DecisionTreeClassifier().fit(
            [[0.0], [1.0]], ["positive", "z_negative"]
        )
        trainer = StreamTrainer(
            model, positive_label="positive", window_size=2
        )

        np.testing.assert_array_equal(
            model.classes_, ["positive", "z_negative"]
        )
        record = trainer.score_chunk(
            [[0.0], [1.0]], ["positive", "z_negative"]
        )

        self.assertEqual(record["cumulative_auc"], 1.0)
        self.assertEqual(record["rolling_auc"], 1.0)
        self.assertEqual(trainer.metrics_result()["cumulative_auc"]["n_samples"], 2)

    def test_missing_positive_class_uses_zero_and_preserves_history(self):
        model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
        trainer = StreamTrainer(model, positive_label=1, window_size=2)

        first = trainer.score_chunk([[0.0], [1.0]], [0, 1])
        self.assertEqual(first["cumulative_auc"], 1.0)

        # The next model contains only the negative class.
        model.fit([[10.0], [11.0]], [0, 0])
        second = trainer.score_chunk([[10.0], [11.0]], [0, 1])

        # Historical scores: negative=[0, 0], positive=[1, 0].
        # Two wins and two ties across four positive/negative pairs.
        self.assertEqual(second["cumulative_auc"], 0.75)
        self.assertEqual(second["rolling_auc"], 0.5)
        self.assertEqual(trainer.metrics_result()["cumulative_auc"]["n_samples"], 4)

    def test_warmup_and_single_class_auc_are_nan(self):
        model = DecisionTreeClassifier(buffer_size=2)
        trainer = StreamTrainer(model, positive_label=1)

        warmup = trainer.score_chunk([[0.0], [1.0]], [0, 1])
        self.assertTrue(warmup["warm_up"])
        self.assertTrue(np.isnan(warmup["cumulative_auc"]))
        self.assertEqual(trainer.metrics_result()["cumulative_auc"]["n_samples"], 0)

        trainer.fit_chunk([[0.0], [1.0]], [0, 1])
        record = trainer.score_chunk([[1.0]], [1])

        self.assertTrue(np.isnan(record["cumulative_auc"]))
        self.assertEqual(trainer.n_samples_evaluated_, 1)
        self.assertEqual(trainer.metrics_result()["cumulative_auc"]["n_positive"], 1)

    def test_auc_disabled_allows_multiclass_but_enabled_rejects_it(self):
        model = DecisionTreeClassifier().fit(
            [[0.0], [1.0], [2.0]], [0, 1, 2]
        )
        disabled = StreamTrainer(model)
        record = disabled.score_chunk([[0.0], [1.0], [2.0]], [0, 1, 2])

        self.assertIsNone(record["cumulative_auc"])
        self.assertEqual(record["cumulative_accuracy"], 1.0)

        enabled = StreamTrainer(model, positive_label=1)
        with self.assertRaisesRegex(ValueError, "non-positive"):
            enabled.score_chunk([[0.0]], [0])

        self.assertEqual(enabled.n_chunks_scored_, 0)
        self.assertEqual(enabled.n_samples_evaluated_, 0)

    def test_auc_failure_rolls_back_all_four_metrics(self):
        model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
        trainer = StreamTrainer(model, positive_label=1)
        trainer.score_chunk([[0.0], [1.0]], [0, 1])

        original_error = RuntimeError("simulated AUC failure")
        with patch.object(RollingROCAUC, "update", side_effect=original_error):
            with self.assertRaises(RuntimeError) as caught:
                trainer.score_chunk([[0.0], [1.0]], [1, 0])

        self.assertIs(caught.exception, original_error)
        self.assertEqual(trainer.n_chunks_scored_, 1)
        result = trainer.metrics_result()
        for name in ("cumulative", "rolling", "cumulative_auc", "rolling_auc"):
            self.assertEqual(result[name]["n_samples"], 2)
        self.assertEqual(result["cumulative"]["accuracy"], 1.0)
        self.assertEqual(result["cumulative_auc"]["auc"], 1.0)

        trainer.score_chunk([[0.0], [1.0]], [1, 0])
        self.assertEqual(trainer.n_chunks_scored_, 2)
        result = trainer.metrics_result()
        self.assertEqual(result["cumulative"]["n_samples"], 4)
        self.assertEqual(result["cumulative_auc"]["n_samples"], 4)
        self.assertEqual(result["cumulative_auc"]["auc"], 0.5)

    def test_categorical_pipeline_probability_alignment(self):
        model = Pipeline([
            ("encoder", OneHotEncoder()),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        model.partial_fit([["a"], ["b"]], [0, 1])
        trainer = StreamTrainer(model, positive_label=0)

        record = trainer.score_chunk([["a"], ["b"]], [0, 1])

        self.assertEqual(record["cumulative_auc"], 1.0)
        self.assertEqual(record["rolling_auc"], 1.0)
        self.assertEqual(model.model_version_, 1)
        self.assertEqual(model.n_samples_buffered_, 0)


if __name__ == "__main__":
    unittest.main()