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

        self.assertAlmostEqual(result, 2.0, places=14)

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

class TestMetricsRegression(unittest.TestCase):
    """Regression tests for numeric stability and input validation."""

    def test_mse_large_finite_mean(self):
        # Each squared error is about 1e308.
        # Their mean is representable even though their sum overflows.
        result = mse([1e154, 1e154], [0.0, 0.0])

        self.assertTrue(np.isfinite(result))
        np.testing.assert_allclose(result, 1e308, rtol=1e-14)

    def test_mse_rejects_mixed_boolean_input(self):
        cases = (
            ([True, 2], [1, 2]),
            ([1, 2], [True, 2]),
            ([[1, False]], [[1, 0]]),
            ([[1, 0]], [[1, False]]),
        )

        for true, pred in cases:
            with self.subTest(true=true, pred=pred):
                with self.assertRaises(TypeError):
                    mse(true, pred)

    def test_confusion_matrix_preserves_large_integer_labels(self):
        # Converting these labels to float64 would change the integer.
        large_label = 2**53 + 1
        true = [large_label, 0.5, large_label]
        pred = [large_label, large_label, 0.5]

        matrix, labels = confusion_matrix(true, pred)

        self.assertEqual(labels.tolist(), [large_label, 0.5])
        np.testing.assert_array_equal(
            matrix,
            [[1, 1], [1, 0]],
        )
        
if __name__ == "__main__":
    unittest.main()