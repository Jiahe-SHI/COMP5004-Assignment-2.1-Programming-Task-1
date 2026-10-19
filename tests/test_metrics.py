"""Tests for evaluation metrics."""

import unittest
import numpy as np

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    f1,
    mse,
)


class TestAccuracy(unittest.TestCase):
    def test_string_labels(self):
        result = accuracy(
            ["cat", "dog"],
            ["cat", "cat"],
        )

        self.assertEqual(result, 0.5)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            accuracy([1, 0, 1], [1, 0])

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            accuracy([], [])
    def test_basic_accuracy(self):
        result = accuracy(
            [1, 0, 1, 1],
            [1, 0, 0, 1],
        )

        self.assertEqual(result, 0.75)

    def test_missing_label(self):
        with self.assertRaises(ValueError):
            accuracy([1, float("nan")], [1, 0])

    def test_2d_input(self):
        with self.assertRaises(ValueError):
            accuracy([[1, 0]], [[1, 0]])

    def test_infinite_label(self):
        with self.assertRaises(ValueError):
            accuracy([1, float("inf")], [1, 0])

class TestConfusionMatrix(unittest.TestCase):
    def test_basic_binary(self):
        matrix, labels = confusion_matrix(
            ["cat", "dog", "cat", "dog"],
            ["cat", "cat", "cat", "dog"],
        )

        self.assertEqual(
            matrix.tolist(),
            [[2, 0], [1, 1]],
        )
        self.assertEqual(
            labels.tolist(),
            ["cat", "dog"],
        )

    def test_explicit_labels(self):
        matrix, labels = confusion_matrix(
            [1, 1, 2],
            [1, 2, 2],
            labels=[2, 1, 3],
        )

        self.assertEqual(
            matrix.tolist(),
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
        )
        self.assertEqual(
            labels.tolist(),
            [2, 1, 3],
        )

    def test_duplicate_explicit_labels(self):
        with self.assertRaises(ValueError):
            confusion_matrix(
                [1, 2],
                [1, 2],
                labels=[1, 1],
            )

    def test_explicit_labels_must_include_observed_labels(self):
        with self.assertRaises(ValueError):
            confusion_matrix(
                [1, 2],
                [1, 2],
                labels=[1],
            )

    def test_invalid_explicit_label_values(self):
        cases = [
            ([0, None], ValueError),
            ([0, float("nan")], ValueError),
            ([0, 1j], TypeError),
        ]

        for labels, error in cases:
            with self.subTest(labels=labels):
                with self.assertRaises(error):
                    confusion_matrix(
                        [0, 0],
                        [0, 0],
                        labels=labels,
                    )

    def test_explicit_label_type_must_match_data(self):
        with self.assertRaises(TypeError):
            confusion_matrix(
                [0, 1],
                [0, 1],
                labels=["zero", "one"],
            )

class TestPrecision(unittest.TestCase):
    def test_binary_precision(self):
        result = precision(
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        )

        self.assertEqual(result, 0.5)

    def test_string_positive_label(self):
        result = precision(
            ["yes", "no", "yes"],
            ["yes", "yes", "no"],
            pos_label="yes",
        )

        self.assertEqual(result, 0.5)

    def test_zero_division(self):
        self.assertEqual(
            precision(
                [1, 0],
                [0, 0],
                zero_division=0,
            ),
            0.0,
        )

        self.assertEqual(
            precision(
                [1, 0],
                [0, 0],
                zero_division=1,
            ),
            1.0,
        )

    def test_binary_rejects_multiclass(self):
        with self.assertRaises(ValueError):
            precision(
                [0, 1, 2],
                [0, 1, 2],
            )


    def test_multiclass_averages(self):
        y_true = [0, 0, 0, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 0]

        self.assertAlmostEqual(
            precision(
                y_true,
                y_pred,
                average="macro",
            ),
            13 / 18,
        )

        self.assertAlmostEqual(
            precision(
                y_true,
                y_pred,
                average="micro",
            ),
            2 / 3,
        )

        self.assertAlmostEqual(
            precision(
                y_true,
                y_pred,
                average="weighted",
            ),
            0.75,
        )

class TestRecall(unittest.TestCase):
    def test_binary_recall(self):
        result = recall(
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        )

        self.assertAlmostEqual(result, 1 / 3)

    def test_multiclass_averages(self):
        y_true = [0, 0, 0, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 0]

        self.assertAlmostEqual(
            recall(
                y_true,
                y_pred,
                average="macro",
            ),
            13 / 18,
        )

        self.assertAlmostEqual(
            recall(
                y_true,
                y_pred,
                average="micro",
            ),
            2 / 3,
        )

        self.assertAlmostEqual(
            recall(
                y_true,
                y_pred,
                average="weighted",
            ),
            2 / 3,
        )

    def test_zero_division(self):
        self.assertEqual(
            recall(
                [0, 0],
                [1, 0],
                pos_label=1,
                zero_division=0,
            ),
            0.0,
        )

        self.assertEqual(
            recall(
                [0, 0],
                [1, 0],
                pos_label=1,
                zero_division=1,
            ),
            1.0,
        )

class TestF1(unittest.TestCase):
    def test_binary_f1(self):
        result = f1(
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        )

        self.assertEqual(result, 0.5)

    def test_multiclass_averages(self):
        y_true = [0, 0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 2, 2]

        self.assertAlmostEqual(
            f1(
                y_true,
                y_pred,
                average="macro",
            ),
            85 / 126,
        )

        self.assertAlmostEqual(
            f1(
                y_true,
                y_pred,
                average="micro",
            ),
            5 / 7,
        )

        self.assertAlmostEqual(
            f1(
                y_true,
                y_pred,
                average="weighted",
            ),
            107 / 147,
        )

class TestMSE(unittest.TestCase):
    def test_basic_1d(self):
        result = mse(
            [1, 2, 3],
            [1, 4, 2],
        )

        self.assertAlmostEqual(result, 5 / 3)

    def test_2d_input(self):
        result = mse(
            [[1, 2], [3, 4]],
            [[1, 0], [5, 4]],
        )

        self.assertEqual(result, 2.0)

    def test_rejects_broadcasting(self):
        with self.assertRaises(ValueError):
            mse(
                [1, 2],
                [[1, 2]],
            )

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            mse([], [])

    def test_invalid_values(self):
        with self.assertRaises(TypeError):
            mse(["one", "two"], ["one", "two"])

        with self.assertRaises(ValueError):
            mse(
                [1, float("nan")],
                [1, 2],
            )

        with self.assertRaises(ValueError):
            mse(
                [1, float("inf")],
                [1, 2],
            )

    def test_noncontiguous_input(self):
        original = np.array(
            [1.0, 99.0, 2.0, 99.0, 3.0, 99.0]
        )
        true = original[::2]
        pred = np.array([1.0, 2.0, 5.0])

        self.assertFalse(true.flags.c_contiguous)
        self.assertAlmostEqual(
            mse(true, pred),
            4 / 3,
        )

class TestClassificationValidation(unittest.TestCase):
    def test_rejects_mixed_label_types(self):
        with self.assertRaises(TypeError):
            accuracy(
                [1, "cat"],
                [1, "cat"],
            )

    def test_true_and_predicted_types_must_match(self):
        with self.assertRaises(TypeError):
            accuracy(
                ["cat", "dog"],
                [1, 2],
            )

    def test_rejects_none_label(self):
        with self.assertRaises(ValueError):
            accuracy(
                [1, None],
                [1, 0],
            )

    def test_rejects_complex_label(self):
        with self.assertRaises(TypeError):
            accuracy(
                [1, 1j],
                [1, 1],
            )

class TestClassificationMetricOptions(unittest.TestCase):
    def test_invalid_average(self):
        for metric in (precision, recall, f1):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(TypeError):
                    metric([0, 1], [0, 1], average=None)

                with self.assertRaises(ValueError):
                    metric(
                        [0, 1],
                        [0, 1],
                        average="invalid",
                    )

    def test_invalid_zero_division(self):
        for metric in (precision, recall, f1):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(TypeError):
                    metric(
                        [0, 1],
                        [0, 1],
                        zero_division=True,
                    )

                with self.assertRaises(ValueError):
                    metric(
                        [0, 1],
                        [0, 1],
                        zero_division=2,
                    )

    def test_unobserved_positive_label(self):
        for metric in (precision, recall, f1):
            with self.subTest(metric=metric.__name__):
                with self.assertRaises(ValueError):
                    metric(
                        [0, 1],
                        [0, 1],
                        pos_label=2,
                    )

    def test_noncontiguous_inputs_unchanged(self):
        true_source = np.array(
            [0, 9, 1, 9, 1, 9]
        )
        pred_source = np.array(
            [0, 9, 0, 9, 1, 9]
        )

        true = true_source[::2]
        pred = pred_source[::2]

        true_before = true.copy()
        pred_before = pred.copy()

        self.assertFalse(true.flags.c_contiguous)
        self.assertFalse(pred.flags.c_contiguous)

        self.assertAlmostEqual(
            accuracy(true, pred),
            2 / 3,
        )

        confusion_matrix(true, pred)
        precision(true, pred, average="macro")
        recall(true, pred, average="macro")
        f1(true, pred, average="macro")

        np.testing.assert_array_equal(
            true,
            true_before,
        )
        np.testing.assert_array_equal(
            pred,
            pred_before,
        )
        
if __name__ == "__main__":
    unittest.main()