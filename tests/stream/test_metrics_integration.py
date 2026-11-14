"""Integration checks for streaming classification metrics and trees."""

from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    RollingClassificationMetrics,
    RollingROCAUC,
    StreamingClassificationMetrics,
    StreamingROCAUC,
)


def pairwise_auc(labels, scores, positive_label):
    """Independent, small-data AUC oracle used only by tests."""
    positive = [
        float(score)
        for label, score in zip(labels, scores)
        if label == positive_label
    ]
    negative = [
        float(score)
        for label, score in zip(labels, scores)
        if label != positive_label
    ]
    if not positive or not negative:
        return float("nan")

    credit = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p in positive
        for n in negative
    )
    return credit / (len(positive) * len(negative))


def positive_scores(tree, X, positive_label):
    """Explicit binary probability alignment for these integration tests.

    An absent positive class receives zero probability. This helper is
    test-local; it does not implement StreamTrainer or Pipeline.
    """
    probabilities = tree.predict_proba(X)
    matches = np.flatnonzero(tree.classes_ == positive_label)
    if matches.size == 0:
        return np.zeros(X.shape[0], dtype=np.float64)
    return probabilities[:, int(matches[0])]


class MetricsIntegrationTests(unittest.TestCase):

    def assert_confusion_matches(self, result, truth, prediction):
        """Build a scalar reference in the result's actual class order."""
        classes = result["classes"].tolist()
        expected = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for actual, predicted in zip(truth, prediction):
            expected[classes.index(actual), classes.index(predicted)] += 1

        np.testing.assert_array_equal(result["confusion_matrix"], expected)
        np.testing.assert_array_equal(result["support"], expected.sum(axis=1))
        self.assertEqual(result["n_samples"], len(truth))
        correct = sum(a == b for a, b in zip(truth, prediction))
        self.assertAlmostEqual(result["accuracy"], correct / len(truth))

    def test_multiclass_counts_across_different_partitions(self):
        truth = np.array([5, 2, 5, 1, 2, 1, 7, 5, 7, 2])
        prediction = np.array([5, 5, 2, 1, 1, 2, 7, 7, 2, 2])
        partitions = (
            [10],
            [1] * 10,
            [2, 5, 1, 2],
        )

        for sizes in partitions:
            with self.subTest(sizes=sizes):
                cumulative = StreamingClassificationMetrics()
                rolling = RollingClassificationMetrics(4)
                end = 0

                for size in sizes:
                    start = end
                    end += size
                    cumulative.update(truth[start:end], prediction[start:end])
                    rolling.update(truth[start:end], prediction[start:end])
                    self.assert_confusion_matches(
                        cumulative.result(),
                        truth[:end],
                        prediction[:end],
                    )
                    self.assert_confusion_matches(
                        rolling.result(),
                        truth[max(0, end - 4):end],
                        prediction[max(0, end - 4):end],
                    )
                    self.assertEqual(rolling.n_samples_seen_, end)

    def test_auc_across_partitions_matches_independent_reference(self):
        truth = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
        scores = np.array([0.2, 0.4, 0.4, 0.8, 0.9,
                           0.1, 0.4, 0.2, 0.7, 0.7])

        for sizes in ([10], [1] * 10, [3, 1, 4, 2]):
            with self.subTest(sizes=sizes):
                cumulative = StreamingROCAUC(positive_label=1)
                rolling = RollingROCAUC(4, positive_label=1)
                end = 0

                for size in sizes:
                    start = end
                    end += size
                    cumulative.update(truth[start:end], scores[start:end])
                    rolling.update(truth[start:end], scores[start:end])

                    expected_cumulative = pairwise_auc(truth[:end], scores[:end], 1)
                    expected_rolling = pairwise_auc(
                        truth[max(0, end - 4):end],
                        scores[max(0, end - 4):end],
                        1,
                    )
                    np.testing.assert_allclose(
                        cumulative.result()["auc"],
                        expected_cumulative,
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        rolling.result()["auc"],
                        expected_rolling,
                        equal_nan=True,
                    )

    def test_tree_positive_class_can_be_first_probability_column(self):
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.array(["a_yes", "a_yes", "z_no", "z_no"])
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X_train, y_train)

        # Separate evaluation rows; the positive class sorts first.
        X_eval = np.array([[0.25], [0.75], [2.25], [2.75]])
        y_eval = np.array(["a_yes", "a_yes", "z_no", "z_no"])
        self.assertEqual(tree.classes_.tolist(), ["a_yes", "z_no"])

        scores = positive_scores(tree, X_eval, "a_yes")
        np.testing.assert_allclose(scores, [1, 1, 0, 0])

        auc = StreamingROCAUC(positive_label="a_yes")
        auc.update(y_eval, scores)
        self.assertEqual(auc.result()["auc"], 1.0)

        wrong_column = StreamingROCAUC(positive_label="a_yes")
        wrong_column.update(y_eval, tree.predict_proba(X_eval)[:, 1])
        self.assertEqual(wrong_column.result()["auc"], 0.0)

        classification = StreamingClassificationMetrics()
        classification.update(y_eval, tree.predict(X_eval))
        self.assertEqual(classification.result()["accuracy"], 1.0)

    def test_tree_refit_with_absent_positive_class(self):
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]),
            np.array(["no", "no", "yes", "yes"]),
        )
        X_eval = np.array([[0.25], [2.75]])
        y_eval = np.array(["no", "yes"])
        first_scores = positive_scores(tree, X_eval, "yes")
        np.testing.assert_allclose(first_scores, [0, 1])

        cumulative = StreamingROCAUC(positive_label="yes")
        rolling = RollingROCAUC(2, positive_label="yes")
        cumulative.update(y_eval, first_scores)
        rolling.update(y_eval, first_scores)

        # Refit publishes a new class table containing only "no".
        tree.fit(
            np.array([[4.0], [5.0]]),
            np.array(["no", "no"]),
        )
        self.assertEqual(tree.classes_.tolist(), ["no"])
        second_scores = positive_scores(tree, X_eval, "yes")
        np.testing.assert_allclose(second_scores, [0, 0])

        cumulative.update(y_eval, second_scores)
        rolling.update(y_eval, second_scores)

        # Historical scores remain [0, 1]; new scores are [0, 0].
        np.testing.assert_allclose(first_scores, [0, 1])
        self.assertAlmostEqual(cumulative.result()["auc"], 0.75)
        self.assertAlmostEqual(rolling.result()["auc"], 0.5)

    def test_deepcopy_snapshots_are_independent(self):
        classification_metrics = (
            StreamingClassificationMetrics(),
            RollingClassificationMetrics(2),
        )
        for metric in classification_metrics:
            metric.update([0, 1], [0, 1])
            snapshot = deepcopy(metric)
            metric.update([0, 1], [1, 0])

            self.assertEqual(snapshot.result()["accuracy"], 1.0)
            self.assertEqual(snapshot.n_samples_seen_, 2)
            self.assertLess(metric.result()["accuracy"], 1.0)

        auc_metrics = (
            StreamingROCAUC(positive_label=1),
            RollingROCAUC(2, positive_label=1),
        )
        for metric in auc_metrics:
            metric.update([0, 1], [0.1, 0.9])
            snapshot = deepcopy(metric)
            metric.update([0, 1], [0.9, 0.1])

            self.assertEqual(snapshot.result()["auc"], 1.0)
            self.assertEqual(snapshot.n_samples_seen_, 2)
            self.assertLess(metric.result()["auc"], 1.0)

    def test_rolling_candidate_failures_preserve_previous_state(self):
        classification = RollingClassificationMetrics(3)
        classification.update([0, 1], [0, 1])
        before_classification = classification.result()

        # Fail after schema validation, during candidate recount.
        with patch.object(
            StreamingClassificationMetrics,
            "update",
            side_effect=RuntimeError("simulated recount failure"),
        ):
            with self.assertRaises(RuntimeError):
                classification.update([0], [1])

        after_classification = classification.result()
        np.testing.assert_array_equal(
            after_classification["confusion_matrix"],
            before_classification["confusion_matrix"],
        )
        self.assertEqual(classification.n_samples_seen_, 2)
        self.assertEqual(after_classification["accuracy"], 1.0)

        auc = RollingROCAUC(3, positive_label=1)
        auc.update([0, 1], [0.1, 0.9])
        before_auc = auc.result()

        # Incoming validation succeeds; window construction then fails.
        with patch(
            "numcompute_stream.metrics.np.concatenate",
            side_effect=MemoryError("simulated allocation failure"),
        ):
            with self.assertRaises(MemoryError):
                auc.update([0], [0.2])

        self.assertEqual(auc.result(), before_auc)

        # Both accumulators remain usable after the failed update.
        classification.update([0], [1])
        auc.update([0], [0.2])
        self.assertEqual(classification.n_samples_seen_, 3)
        self.assertEqual(auc.n_samples_seen_, 3)


if __name__ == "__main__":
    unittest.main()