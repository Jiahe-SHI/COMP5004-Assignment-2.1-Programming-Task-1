import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from numcompute.sort_search import (
    binary_search, multi_key_sort, quickselect, stable_sort, topk,
)


class TestStableSort(unittest.TestCase):

    def test_ascending_and_original_indices(self):
        values, indices = stable_sort([3, 1, 3, 2, 1], return_indices=True)
        assert_array_equal(values, [1, 1, 2, 3, 3])
        assert_array_equal(indices, [1, 4, 3, 0, 2])

    def test_descending_preserves_tie_order(self):
        values, indices = stable_sort([3, 1, 3, 2, 1], True, True)
        assert_array_equal(values, [3, 3, 2, 1, 1])
        assert_array_equal(indices, [0, 2, 3, 1, 4])

    def test_empty_singleton_and_all_equal(self):
        self.assertEqual(stable_sort([]).shape, (0,))
        assert_array_equal(stable_sort([7]), [7])
        for descending in (False, True):
            _, indices = stable_sort([4, 4, 4], descending, True)
            assert_array_equal(indices, [0, 1, 2])

    def test_infinities_and_signed_zeros(self):
        values = np.array([0.0, -0.0, np.inf, -np.inf, 0.0])
        _, indices = stable_sort(values, True, True)
        assert_array_equal(indices, [2, 0, 1, 4, 3])

    def test_integer_precision_and_dtype(self):
        values = np.array([2**63 + 1, 2**63, 2**64 - 1], dtype=np.uint64)
        result = stable_sort(values)
        self.assertEqual(result.dtype, values.dtype)
        assert_array_equal(result, values[[1, 0, 2]])

    def test_non_contiguous_and_independent_output(self):
        base = np.array([8, 99, 2, 99, 6, 99, 2])
        before = base.copy()
        view = base[::2]
        self.assertFalse(view.flags.c_contiguous)
        result = stable_sort(view)
        assert_array_equal(result, [2, 2, 6, 8])
        result[:] = -1
        assert_array_equal(base, before)

    def test_invalid_data_shapes_and_options(self):
        for values in ([True, 1], ["1", "2"], [1j], np.array([1], dtype=object)):
            with self.subTest(values=values), self.assertRaises(TypeError):
                stable_sort(values)
        for values in (1, [[1, 2]], [[1], [2, 3]], [np.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stable_sort(values)
        for option in ({"descending": 1}, {"return_indices": "yes"}):
            with self.assertRaises(TypeError):
                stable_sort([1], **option)


class TestMultiKeySort(unittest.TestCase):

    def test_priority_and_stable_row_ties(self):
        X = np.array([[2, 1, 0], [1, 2, 1], [1, 1, 2],
                      [2, 1, 3], [1, 1, 4]])
        rows, indices = multi_key_sort(X, [0, 1], True)
        assert_array_equal(indices, [2, 4, 1, 0, 3])
        assert_array_equal(rows, X[[2, 4, 1, 0, 3]])
        _, swapped = multi_key_sort(X, [1, 0], True)
        assert_array_equal(swapped, [2, 4, 0, 3, 1])

    def test_empty_rows_and_single_key(self):
        self.assertEqual(multi_key_sort(np.empty((0, 2)), [1]).shape, (0, 2))
        assert_array_equal(multi_key_sort([[2, 7], [1, 8]], [0]),
                           [[1, 8], [2, 7]])

    def test_strides_integer_extremes_and_input_preservation(self):
        base = np.array([[2**63 + 1, 0, 4, 0], [2**63, 0, 5, 0]], dtype=np.uint64)
        before = base.copy()
        rows = multi_key_sort(base[:, ::2], [0])
        self.assertEqual(rows.dtype, np.dtype("uint64"))
        assert_array_equal(rows, base[[1, 0], ::2])
        rows[:] = 0
        assert_array_equal(base, before)

    def test_infinity_allowed_nan_rejected_anywhere(self):
        rows = multi_key_sort([[np.inf, 0], [-np.inf, 0]], [0])
        assert_array_equal(rows[:, 0], [-np.inf, np.inf])
        with self.assertRaises(ValueError):
            multi_key_sort([[1, np.nan]], [0])

    def test_invalid_columns(self):
        for columns in ([], [0, 0], [-1], [2], [[0]]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                multi_key_sort([[1, 2]], columns)
        for columns in ([0.0], [True], [0, True], ["0"]):
            with self.subTest(columns=columns), self.assertRaises(TypeError):
                multi_key_sort([[1, 2]], columns)

    def test_invalid_table_or_flag(self):
        with self.assertRaises(ValueError):
            multi_key_sort([1, 2], [0])
        with self.assertRaises(TypeError):
            multi_key_sort([["a"]], [0])
        with self.assertRaises(TypeError):
            multi_key_sort([[1]], [0], return_indices=1)


class TestTopK(unittest.TestCase):

    def test_largest_boundary_ties(self):
        values, indices = topk([5, 9, 9, 9, 4], 2)
        assert_array_equal(values, [9, 9])
        assert_array_equal(indices, [1, 2])

    def test_smallest_boundary_ties(self):
        values, indices = topk([3, 1, 1, 1, -2], 3, largest=False)
        assert_array_equal(values, [-2, 1, 1])
        assert_array_equal(indices, [4, 1, 2])

    def test_zero_full_and_values_only(self):
        values, indices = topk([], 0)
        self.assertEqual(values.shape, (0,))
        self.assertEqual(indices.dtype, np.dtype(np.intp))
        assert_array_equal(topk([2, 1, 2], 3, return_indices=False), [2, 2, 1])
        assert_array_equal(topk([2, 1, 2], 3)[1], [0, 2, 1])
        assert_array_equal(topk([2], 0, return_indices=False), [])

    def test_all_equal_and_infinities(self):
        for largest in (False, True):
            assert_array_equal(topk(np.ones(100), 4, largest)[1], [0, 1, 2, 3])
        assert_array_equal(topk([np.inf, -np.inf, np.inf, 0], 3)[1], [0, 2, 3])

    def test_integer_extremes_and_non_contiguous(self):
        values = np.array([-(2**63), 2**63 - 1, 0, 2**63 - 1], dtype=np.int64)
        assert_array_equal(topk(values, 3)[1], [1, 3, 2])
        source = np.array([2**64 - 1, 0, 2**63, 0, 2**63 + 1], dtype=np.uint64)
        before = source.copy()
        selected, indices = topk(source[::2], 2)
        assert_array_equal(indices, [0, 2])
        self.assertEqual(selected.dtype, source.dtype)
        selected[:] = 0
        assert_array_equal(source, before)

    def test_seeded_reference_all_k_and_tie_orders(self):
        rng = np.random.default_rng(42)
        for n in (1, 2, 7, 31):
            data = rng.integers(-3, 4, n)
            for largest in (False, True):
                expected = sorted(range(n), key=lambda i: (-int(data[i]) if largest
                                                          else int(data[i]), i))
                for k in range(n + 1):
                    with self.subTest(n=n, largest=largest, k=k):
                        values, indices = topk(data, k, largest)
                        assert_array_equal(indices, expected[:k])
                        assert_array_equal(values, data[expected[:k]])

    def test_invalid_parameters_and_nan_even_for_zero_k(self):
        for k in (True, 1.0, "1"):
            with self.assertRaises(TypeError):
                topk([1, 2], k)
        for values, k in (([1], -1), ([1], 2), ([], 1), ([np.nan], 0)):
            with self.assertRaises(ValueError):
                topk(values, k)
        with self.assertRaises(TypeError):
            topk([1], 1, largest=1)
        with self.assertRaises(TypeError):
            topk([1], 1, return_indices=None)
        with self.assertRaises(ValueError):
            topk([[1]], 1)


class TestQuickselect(unittest.TestCase):

    def test_known_values_each_position(self):
        data = [7, 2, 9, 2, -1, 4]
        for k, expected in enumerate([-1, 2, 2, 4, 7, 9]):
            self.assertEqual(quickselect(data, k), expected)

    def test_singleton_equal_sorted_and_reverse_sorted(self):
        for data in ([5], [3] * 1000, list(range(100)), list(range(99, -1, -1))):
            for k in (0, len(data) // 2, len(data) - 1):
                self.assertEqual(quickselect(data, k), sorted(data)[k])

    def test_infinities_and_integer_precision(self):
        self.assertEqual(quickselect([np.inf, -np.inf, 0, np.inf], 0), -np.inf)
        self.assertEqual(quickselect([np.inf, -np.inf, 0, np.inf], 3), np.inf)
        data = np.array([2**64 - 1, 2**63 + 1, 2**63], dtype=np.uint64)
        self.assertEqual(quickselect(data, 1), np.uint64(2**63 + 1))

    def test_manual_algorithm_and_input_preservation(self):
        base = np.array([8, 99, 2, 99, 6, 99, 1])
        before = base.copy()
        with patch("numpy.partition", side_effect=AssertionError("partition")), \
             patch("numpy.argpartition", side_effect=AssertionError("argpartition")), \
             patch("numpy.sort", side_effect=AssertionError("sort")), \
             patch("numpy.argsort", side_effect=AssertionError("argsort")):
            self.assertEqual(quickselect(base[::2], 2), 6)
        assert_array_equal(base, before)

    def test_seeded_independent_reference(self):
        rng = np.random.default_rng(19)
        for n in (3, 17, 100, 511):
            values = rng.integers(-20, 21, n)
            reference = sorted(values.tolist())
            for k in (0, 1, n // 3, n // 2, n - 1):
                self.assertEqual(quickselect(values, np.int64(k)), reference[k])

    def test_invalid_parameters(self):
        for values, k in (([], 0), ([1], -1), ([1], 1), ([np.nan], 0), ([[1]], 0)):
            with self.assertRaises(ValueError):
                quickselect(values, k)
        for k in (True, 0.0, "0"):
            with self.assertRaises(TypeError):
                quickselect([1], k)
        with self.assertRaises(TypeError):
            quickselect(["1"], 0)


class TestBinarySearch(unittest.TestCase):

    def test_first_duplicate_and_insertion_points(self):
        data = np.array([1, 2, 2, 2, 5])
        for target, expected in ((2, (1, True)), (0, (0, False)),
                                 (3, (4, False)), (6, (5, False))):
            self.assertEqual(binary_search(data, target), expected)

    def test_empty_singleton_and_all_equal(self):
        self.assertEqual(binary_search([], 1), (0, False))
        self.assertEqual(binary_search([3], 3), (0, True))
        self.assertEqual(binary_search(np.ones(1000), 1), (0, True))

    def test_infinities_and_non_contiguous_input(self):
        data = np.array([-np.inf, 0, 1, 2, 3, np.inf])[::2]
        self.assertEqual(binary_search(data, 1), (1, True))
        self.assertEqual(binary_search([-np.inf, 0, np.inf], np.inf), (2, True))

    def test_large_and_mixed_signed_unsigned_targets(self):
        data = np.array([2**63, 2**63 + 1, 2**64 - 1], dtype=np.uint64)
        self.assertEqual(binary_search(data, 2**63 + 1), (1, True))
        self.assertEqual(binary_search(data, -1), (0, False))
        self.assertEqual(binary_search(np.array([-1, 0], dtype=np.int64),
                                       np.uint64(2**64 - 1)), (2, False))

    def test_does_not_scan_or_copy_array(self):
        values = np.arange(100_000)
        real_isnan = np.isnan
        calls = []

        def scalar_isnan(value):
            self.assertEqual(np.ndim(value), 0)
            calls.append(value)
            return real_isnan(value)

        with patch("numcompute.sort_search.np.isnan", side_effect=scalar_isnan), \
             patch("numcompute.sort_search._as_float_array",
                   side_effect=AssertionError("No array conversion/copy")):
            self.assertEqual(binary_search(values, 71_234), (71_234, True))
        self.assertEqual(len(calls), 1)

    def test_reference_searchsorted_and_original_input(self):
        values = np.arange(-50, 51, 2)
        before = values.copy()
        for target in np.arange(-55, 56, 0.5):
            index = int(np.searchsorted(values, target, side="left"))
            expected = (index, bool(index < values.size and values[index] == target))
            self.assertEqual(binary_search(values, target), expected)
        assert_array_equal(values, before)

    def test_invalid_shape_types_and_nan_target(self):
        for values, target in (([[1]], 1), ([1], [1]), ([1], np.nan)):
            with self.assertRaises(ValueError):
                binary_search(values, target)
        for values, target in ((["1"], 1), ([1], True), ([1], "1"), ([1], 1j)):
            with self.assertRaises(TypeError):
                binary_search(values, target)


if __name__ == "__main__":
    unittest.main()
