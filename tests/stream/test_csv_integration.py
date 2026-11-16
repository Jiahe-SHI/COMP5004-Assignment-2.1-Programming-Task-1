"""Integration tests for CSV input and prequential learning."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    Pipeline,
    StandardScaler,
    StreamTrainer,
    iter_csv_chunks,
)


class TestCSVIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "stream.csv"

    def write_csv(self, text):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def make_trainer(self, buffer_size=3, positive_label=1):
        return StreamTrainer(
            Pipeline([
                ("imputer", Imputer()),
                ("scaler", StandardScaler()),
                ("tree", DecisionTreeClassifier(
                    max_depth=3,
                    buffer_size=buffer_size,
                    random_state=2026,
                )),
            ]),
            window_size=3,
            positive_label=positive_label,
        )

    def test_numeric_csv_preserves_evaluation_and_buffer_semantics(self):
        self.write_csv(
            "x,target\n"
            "0,0\n1,0\n2,1\n3,1\n"
            ",0\n5,1\n6,1\n7,0\n8,1\n"
        )
        trainer = self.make_trainer()
        model = trainer.model
        records = []
        truths = []
        predictions = []
        training_sizes = []

        for X, y in iter_csv_chunks(
            self.path,
            target_column="target",
            label_dtype="int",
            chunk_size=2,
        ):
            before = (
                model.model_version_,
                model.n_samples_buffered_,
                model.n_retrains_,
            )
            if model.is_fitted_:
                truths.extend(y.tolist())
                predictions.extend(model.predict(X).tolist())

            record = trainer.score_chunk(X, y)
            records.append(record)

            self.assertEqual(
                (
                    model.model_version_,
                    model.n_samples_buffered_,
                    model.n_retrains_,
                ),
                before,
            )
            self.assertEqual(record["model_version"], before[0])

            evaluated = trainer.n_samples_evaluated_
            trainer.fit_chunk(X, y)
            self.assertEqual(trainer.n_samples_evaluated_, evaluated)

            if model.n_retrains_ > before[2]:
                training_sizes.append(model.n_samples_fit_)

        self.assertEqual(
            [r["warm_up"] for r in records],
            [True, True, False, False, False],
        )
        self.assertEqual(
            [r["model_version"] for r in records],
            [0, 0, 1, 1, 2],
        )
        self.assertTrue(
            all(np.isnan(r["chunk_accuracy"]) for r in records[:2])
        )
        self.assertEqual(trainer.n_samples_scored_, 9)
        self.assertEqual(trainer.n_samples_evaluated_, 5)
        self.assertEqual(training_sizes, [4, 4])
        self.assertEqual(model.n_retrains_, 2)
        self.assertEqual(model.n_samples_buffered_, 1)

        truth = np.asarray(truths)
        prediction = np.asarray(predictions)
        result = trainer.metrics_result()

        self.assertAlmostEqual(
            result["cumulative"]["accuracy"],
            float(np.mean(truth == prediction)),
        )
        self.assertAlmostEqual(
            result["rolling"]["accuracy"],
            float(np.mean(truth[-3:] == prediction[-3:])),
        )
        self.assertEqual(result["cumulative_auc"]["n_samples"], 5)
        self.assertEqual(result["rolling_auc"]["n_samples"], 3)

        logs = trainer.logs_
        self.assertEqual([row["event"] for row in logs], ["score", "fit"] * 5)
        self.assertTrue(all(row["success"] for row in logs))

    def test_string_labels_flow_from_csv_to_metrics(self):
        self.write_csv(
            "x,target\n"
            "0,no\n1,yes\n"
            "0,yes\n1,no\n"
        )
        trainer = self.make_trainer(buffer_size=2, positive_label="yes")
        records = []

        for X, y in iter_csv_chunks(
            self.path,
            target_column="target",
            label_dtype="str",
            chunk_size=2,
        ):
            records.append(trainer.score_chunk(X, y))
            trainer.fit_chunk(X, y)

        self.assertEqual([r["warm_up"] for r in records], [True, False])
        self.assertEqual(trainer.n_samples_evaluated_, 2)
        result = trainer.metrics_result()

        # The second chunk reverses the labels learned from the first.
        # Evaluation must use the model from before the second fit.
        self.assertEqual(result["cumulative"]["accuracy"], 0.0)
        self.assertEqual(result["cumulative_auc"]["auc"], 0.0)
        self.assertEqual(trainer.model.n_retrains_, 2)
        np.testing.assert_array_equal(
            trainer.model.predict([[0.0], [1.0]]),
            ["yes", "no"],
        )

    def test_csv_error_does_not_process_an_incomplete_chunk(self):
        self.write_csv(
            "x,target\n"
            "0,0\n1,1\n"
            "2,0\nbad,1\n"
        )
        trainer = self.make_trainer(buffer_size=2)
        stream = iter_csv_chunks(
            self.path,
            target_column="target",
            label_dtype="int",
            chunk_size=2,
        )
        try:
            with self.assertRaisesRegex(ValueError, "line 5"):
                for X, y in stream:
                    trainer.score_chunk(X, y)
                    trainer.fit_chunk(X, y)
        finally:
            stream.close()

        # Only the first complete chunk reached the trainer.
        self.assertEqual(trainer.n_chunks_scored_, 1)
        self.assertEqual(trainer.n_samples_scored_, 2)
        self.assertEqual(trainer.n_samples_evaluated_, 0)
        self.assertEqual(trainer.model.n_retrains_, 1)
        self.assertEqual(trainer.model.n_samples_buffered_, 0)
        self.assertEqual(
            [row["event"] for row in trainer.logs_],
            ["score", "fit"],
        )


if __name__ == "__main__":
    unittest.main()