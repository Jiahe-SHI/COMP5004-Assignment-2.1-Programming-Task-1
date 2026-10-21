"""Tests for column selection, validation and fitted state."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from numcompute.preprocessing import ColumnSelector


class TestColumnSelector(unittest.TestCase):

    def test_column_order_and_single_column_shape(self):
        X = np.array([[1, 2, 3], [4, 5, 6]])

        result = ColumnSelector([2, 0]).fit_transform(X)
        assert_array_equal(result, [[3, 1], [6, 4]])

        single = ColumnSelector([1]).fit_transform(X)
        self.assertEqual(single.shape, (2, 1))
        assert_array_equal(single, [[2], [5]])

    def test_dtype_preservation_and_numeric_conversion(self):
        X = np.array([[1.0, "red"], [2.0, "blue"]], dtype=object)

        categories = ColumnSelector([1]).fit_transform(X)
        self.assertEqual(categories.dtype, np.dtype(object))
        assert_array_equal(categories, [["red"], ["blue"]])

        numbers = ColumnSelector([0], dtype=np.float64).fit_transform(X)
        self.assertEqual(numbers.dtype, np.dtype("float64"))
        assert_array_equal(numbers, [[1.0], [2.0]])

    def test_noncontiguous_input_and_independent_output(self):
        source = np.arange(24).reshape(4, 6)
        before = source.copy()
        X = source[:, ::2]

        result = ColumnSelector([2, 0]).fit_transform(X)

        assert_array_equal(result, X[:, [2, 0]])
        self.assertFalse(np.shares_memory(result, source))

        result[:] = -1
        assert_array_equal(source, before)

    def test_fit_returns_self_and_records_feature_count(self):
        selector = ColumnSelector([0])
        self.assertIs(selector.fit([[1, 2]]), selector)
        self.assertEqual(selector.n_features_in_, 2)

    def test_fit_transform_equivalence(self):
        X = [[1, 2], [3, 4]]

        combined = ColumnSelector([1, 0]).fit_transform(X)
        separate = ColumnSelector([1, 0]).fit(X).transform(X)

        assert_array_equal(combined, separate)

    def test_transform_before_fit(self):
        with self.assertRaises(RuntimeError):
            ColumnSelector([0]).transform([[1, 2]])

    def test_invalid_column_types(self):
        for columns in ([True], [0, False], [0.0], ["0"]):
            with self.subTest(columns=columns):
                with self.assertRaises(TypeError):
                    ColumnSelector(columns)

    def test_invalid_column_values(self):
        for columns in ([], [0, 0], [-1], [[0]]):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    ColumnSelector(columns)

        with self.assertRaises(ValueError):
            ColumnSelector([2]).fit([[1, 2]])

    def test_numpy_integer_columns(self):
        selector = ColumnSelector(np.array([1, 0], dtype=np.int64))
        assert_array_equal(selector.fit_transform([[3, 4]]), [[4, 3]])

    def test_invalid_input_dimensions(self):
        for X in (1, [1, 2], np.zeros((1, 2, 3))):
            with self.subTest(shape=np.shape(X)):
                with self.assertRaises(ValueError):
                    ColumnSelector([0]).fit(X)

        selector = ColumnSelector([0]).fit([[1, 2]])
        with self.assertRaises(ValueError):
            selector.transform([1, 2])

    def test_empty_fit_rejected_empty_transform_allowed(self):
        for X in (np.empty((0, 2)), np.empty((2, 0))):
            with self.subTest(shape=X.shape):
                with self.assertRaises(ValueError):
                    ColumnSelector([0]).fit(X)

        selector = ColumnSelector([1]).fit([[1, 2]])
        result = selector.transform(np.empty((0, 2)))
        self.assertEqual(result.shape, (0, 1))

    def test_wrong_feature_count(self):
        selector = ColumnSelector([0]).fit([[1, 2]])

        with self.assertRaises(ValueError):
            selector.transform([[1, 2, 3]])

    def test_parameter_change_requires_refit(self):
        columns = [0]
        selector = ColumnSelector(columns).fit([[1, 2]])

        columns[0] = 1
        with self.assertRaises(RuntimeError):
            selector.transform([[1, 2]])

        selector.fit([[1, 2]])
        assert_array_equal(selector.transform([[3, 4]]), [[4]])

        selector.dtype = np.float64
        with self.assertRaises(RuntimeError):
            selector.transform([[3, 4]])

        result = selector.fit_transform([[3, 4]])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_failed_refit_preserves_previous_state(self):
        selector = ColumnSelector([1], dtype=float).fit([[1, 2]])

        with self.assertRaises(ValueError):
            selector.fit([[1, "invalid", 3]])

        self.assertEqual(selector.n_features_in_, 2)
        assert_array_equal(selector.transform([[4, 5]]), [[5.0]])

    def test_refit_updates_feature_count(self):
        selector = ColumnSelector([0]).fit([[1, 2]])
        selector.fit([[3, 4, 5]])

        self.assertEqual(selector.n_features_in_, 3)
        assert_array_equal(selector.transform([[6, 7, 8]]), [[6]])

    def test_invalid_conversion_during_transform(self):
        selector = ColumnSelector([0], dtype=float).fit([[1]])

        with self.assertRaises(ValueError):
            selector.transform([["invalid"]])


if __name__ == "__main__":
    unittest.main()