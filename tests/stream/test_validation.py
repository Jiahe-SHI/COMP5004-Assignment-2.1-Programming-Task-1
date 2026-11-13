"""Tests for shared feature, label and tree parameter validation."""

import unittest

import numpy as np

from numcompute_stream._validation import (
    validate_integer,
    validate_numeric_X,
    validate_y,
    validate_X_y,
    validate_tree_parameters,
)


class TestNumericFeatures(unittest.TestCase):
    def test_accepts_numeric_matrix(self):
        result = validate_numeric_X([[1, 2], [3, 4]])
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_rejects_invalid_dimensions(self):
        for X in (1, [1, 2], np.zeros((1, 2, 3))):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    validate_numeric_X(X)

    def test_rejects_empty_training_inputs(self):
        for shape in ((0, 2), (2, 0), (0, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    validate_numeric_X(np.empty(shape))

    def test_allows_empty_prediction_with_matching_features(self):
        result = validate_numeric_X(
            np.empty((0, 2)), allow_empty=True, n_features=2
        )
        self.assertEqual(result.shape, (0, 2))

    def test_rejects_prediction_feature_mismatch(self):
        with self.assertRaises(ValueError):
            validate_numeric_X(
                np.empty((0, 3)), allow_empty=True, n_features=2
            )

    def test_nan_requires_explicit_permission(self):
        X = np.array([[1.0, np.nan]])
        with self.assertRaises(ValueError):
            validate_numeric_X(X)
        result = validate_numeric_X(X, allow_nan=True)
        self.assertTrue(np.isnan(result[0, 1]))

    def test_infinity_is_always_rejected(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_numeric_X([[value]], allow_nan=True)

    def test_rejects_unsupported_feature_types(self):
        inputs = (
            [["1"]],
            [[True]],
            [[1 + 2j]],
            np.array([[1]], dtype=object),
        )
        for X in inputs:
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    validate_numeric_X(X)

    def test_does_not_modify_readonly_input(self):
        X = np.array([[1.0, 2.0]])
        X.setflags(write=False)
        result = validate_numeric_X(X)
        np.testing.assert_array_equal(result, [[1.0, 2.0]])
        self.assertFalse(X.flags.writeable)


class TestLabels(unittest.TestCase):
    def test_accepts_integer_and_string_labels(self):
        for y in ([2, 1, 2], ["cat", "dog", "cat"]):
            with self.subTest(y=y):
                np.testing.assert_array_equal(validate_y(y), y)

    def test_preserves_large_integer_list(self):
        values = [2**53 + 1, 2**53 + 2, 2**80]
        result = validate_y(values)
        self.assertEqual(result.tolist(), values)

    def test_preserves_unsigned_integer_array(self):
        values = np.array([2**63, 2**64 - 1], dtype=np.uint64)
        result = validate_y(values)
        self.assertEqual(result.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(result, values)

    def test_rejects_mixed_labels_before_string_coercion(self):
        for y in ([1, "cat"], [1, True], [1, 2.0]):
            with self.subTest(y=y):
                with self.assertRaises(TypeError):
                    validate_y(y)

    def test_rejects_invalid_label_types(self):
        inputs = (
            [True, False],
            [1.0, 2.0],
            [1, None],
            [1, np.nan],
            [b"cat", b"dog"],
            np.array([1, "cat"], dtype=object),
        )
        for y in inputs:
            with self.subTest(y=y):
                with self.assertRaises(TypeError):
                    validate_y(y)

    def test_rejects_invalid_label_dimensions(self):
        for y in (1, [[1], [2]]):
            with self.subTest(y=y):
                with self.assertRaises(ValueError):
                    validate_y(y)

    def test_empty_labels_require_permission(self):
        with self.assertRaises(ValueError):
            validate_y([])
        self.assertEqual(validate_y([], allow_empty=True).shape, (0,))

    def test_rejects_sample_count_mismatch(self):
        with self.assertRaises(ValueError):
            validate_X_y([[1], [2]], [0])

    def test_validates_matching_training_pair(self):
        X, y = validate_X_y([[1], [2]], ["a", "b"])
        self.assertEqual(X.shape, (2, 1))
        self.assertEqual(y.shape, (2,))


class TestTreeParameters(unittest.TestCase):
    @staticmethod
    def check_parameters(**overrides):
        parameters = {
            "max_depth": None,
            "min_samples_split": 2,
            "max_features": None,
            "buffer_size": 500,
            "criterion": "gini",
            "random_state": None,
            "n_features": 3,
        }
        parameters.update(overrides)
        validate_tree_parameters(**parameters)

    def test_accepts_supported_parameter_values(self):
        for feature_option in (None, 1, np.int64(3), "sqrt", "log2"):
            with self.subTest(max_features=feature_option):
                self.check_parameters(
                    max_depth=0,
                    max_features=feature_option,
                    criterion="entropy",
                    random_state=0,
                )

    def test_integer_validator_rejects_bool_and_float(self):
        for value in (True, np.bool_(False), 2.0, "2"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    validate_integer(value, "value")

    def test_rejects_out_of_range_parameters(self):
        invalid = (
            {"max_depth": -1},
            {"min_samples_split": 1},
            {"buffer_size": 0},
            {"max_features": 0},
            {"max_features": 4},
            {"random_state": -1},
            {"n_features": 0},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.check_parameters(**parameters)

    def test_rejects_invalid_parameter_types(self):
        invalid = (
            {"max_depth": True},
            {"min_samples_split": 2.0},
            {"buffer_size": False},
            {"max_features": 0.5},
            {"random_state": 1.0},
            {"criterion": None},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                with self.assertRaises(TypeError):
                    self.check_parameters(**parameters)

    def test_rejects_unknown_named_options(self):
        for parameters in (
            {"criterion": "unknown"},
            {"max_features": "auto"},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.check_parameters(**parameters)


if __name__ == "__main__":
    unittest.main()