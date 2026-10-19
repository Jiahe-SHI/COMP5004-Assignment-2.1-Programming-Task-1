import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute.rank import percentile, rank
from numcompute.sort_search import binary_search, stable_sort, topk
from numcompute.stats import quantile


class TestRank(unittest.TestCase):

    def test_average_ties(self):
        assert_array_equal(rank([40, 10, 10, 20]), [4, 1.5, 1.5, 3])

    def test_dense_ties(self):
        assert_array_equal(rank([40, 10, 10, 20], "dense"), [3, 1, 1, 2])

    def test_ordinal_uses_original_tie_order(self):
        assert_array_equal(rank([40, 10, 10, 20], "ordinal"), [4, 1, 2, 3])

    def test_empty_singleton_and_all_equal(self):
        for method, expected in (("average", [2, 2, 2]), ("dense", [1, 1, 1]),
                                 ("ordinal", [1, 2, 3])):
            assert_array_equal(rank([5, 5, 5], method), expected)
            self.assertEqual(rank([], method).shape, (0,))
            assert_array_equal(rank([5], method), [1])

    def test_nan_propagate_omit_and_raise(self):
        data = [3, np.nan, 1, 1]
        self.assertTrue(np.isnan(rank(data)).all())
        for method, expected in (("average", [3, np.nan, 1.5, 1.5]),
                                 ("dense", [2, np.nan, 1, 1]),
                                 ("ordinal", [3, np.nan, 1, 2])):
            assert_allclose(rank(data, method, "omit"), expected, equal_nan=True)
        with self.assertRaises(ValueError):
            rank(data, nan_policy="raise")

    def test_all_nan_omit_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.isnan(rank([np.nan, np.nan], nan_policy="omit")).all())

    def test_infinities_signed_zero_and_large_integer_ties(self):
        assert_array_equal(rank([np.inf, -np.inf, 0, -0.0, np.inf]), [4.5, 1, 2.5, 2.5, 4.5])
        data = np.array([2**63 + 1, 2**63, 2**63 + 1], dtype=np.uint64)
        assert_array_equal(rank(data), [2.5, 1, 2.5])

    def test_non_contiguous_dtype_and_independent_output(self):
        base = np.array([4, 99, 1, 99, 1, 99, 2])
        before = base.copy()
        result = rank(base[::2])
        self.assertEqual(result.dtype, np.dtype("float64"))
        assert_array_equal(result, [4, 1.5, 1.5, 3])
        result[:] = -1
        assert_array_equal(base, before)

    def test_seeded_pairwise_reference(self):
        rng = np.random.default_rng(17)
        values = rng.integers(-5, 6, 60)
        for method in ("average", "dense", "ordinal"):
            expected = []
            for i, value in enumerate(values):
                smaller = sum(int(v < value) for v in values)
                if method == "average":
                    equal = sum(int(v == value) for v in values)
                    expected.append(smaller + (equal + 1) / 2)
                elif method == "dense":
                    expected.append(1 + len({int(v) for v in values if v < value}))
                else:
                    prior_ties = sum(int(v == value) for v in values[:i])
                    expected.append(1 + smaller + prior_ties)
            assert_array_equal(rank(values, method), expected)

    def test_invalid_data_shapes_and_options(self):
        for values in ([True, 1], ["1"], [1j], np.array([1], dtype=object)):
            with self.assertRaises(TypeError):
                rank(values)
        for values in (1, [[1]], [[1], [1, 2]]):
            with self.assertRaises(ValueError):
                rank(values)
        for kwargs in ({"method": "min"}, {"nan_policy": "ignore"}):
            with self.assertRaises(ValueError):
                rank([1], **kwargs)
        for kwargs in ({"method": 1}, {"nan_policy": None}):
            with self.assertRaises(TypeError):
                rank([1], **kwargs)


class TestPercentile(unittest.TestCase):

    def test_four_hand_calculated_methods(self):
        for method, expected in (("linear", 7.5), ("lower", 0),
                                 ("higher", 10), ("midpoint", 5)):
            self.assertEqual(percentile([0, 10, 20, 30], 25, method), expected)

    def test_boundaries_unsorted_q_and_single_value(self):
        assert_array_equal(percentile([30, 0, 20, 10], [100, 0, 50]), [30, 0, 15])
        assert_array_equal(percentile([7], [0, 33, 100]), [7, 7, 7])

    def test_axis_shapes_scalar_and_vector_q(self):
        X = np.array([[0, 10, 20], [20, 30, 40]])
        assert_array_equal(percentile(X, 50, axis=0), [10, 20, 30])
        assert_array_equal(percentile(X, [0, 100], axis=-1), [[0, 20], [20, 40]])
        self.assertIsInstance(percentile(X, 50), np.float64)
        self.assertEqual(percentile(X, [50]).shape, (1,))
        self.assertEqual(percentile(X, [25, 75], axis=0).shape, (2, 3))

    def test_nan_policy_per_slice_and_all_nan(self):
        X = [[1, np.nan, np.nan], [3, 8, np.nan]]
        assert_allclose(percentile(X, 50, axis=0), [2, np.nan, np.nan], equal_nan=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert_allclose(percentile(X, 50, axis=0, nan_policy="omit"),
                            [2, 8, np.nan], equal_nan=True)
        with self.assertRaises(ValueError):
            percentile(X, 50, nan_policy="raise")

    def test_empty_reductions_and_empty_q(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(np.isnan(percentile([], 50)))
            result = percentile(np.empty((2, 0)), [0, 100], axis=1)
            self.assertEqual(result.shape, (2, 2))
            self.assertTrue(np.isnan(result).all())
            self.assertEqual(percentile(np.empty((0, 3)), 50, axis=1).shape, (0,))
            self.assertEqual(percentile([[1, 2], [3, 4]], [], axis=0).shape, (0, 2))

    def test_interpolation_alias_and_conflict(self):
        self.assertEqual(percentile([0, 10], 25, interpolation="higher"), 10)
        self.assertEqual(percentile([0, 10], 25, "lower", interpolation="lower"), 0)
        with self.assertRaises(ValueError):
            percentile([0, 10], 25, "lower", interpolation="higher")
        with self.assertRaises(TypeError):
            percentile([0, 10], 25, interpolation=1)

    def test_numpy_reference_multiple_axes_and_methods(self):
        X = np.random.default_rng(3).normal(size=(4, 5, 6))
        for axis in (None, 0, 1, 2, -1):
            for method in ("linear", "lower", "higher", "midpoint"):
                for q in (31, [0, 17, 50, 99, 100]):
                    with self.subTest(axis=axis, method=method, q=q):
                        assert_allclose(percentile(X, q, method, axis),
                                        np.percentile(X, q, axis=axis, method=method))

    def test_shared_quantile_semantics(self):
        X = [[1, np.nan, 3], [5, 7, 9]]
        for method in ("linear", "lower", "higher", "midpoint"):
            for policy in ("omit", "propagate"):
                assert_allclose(percentile(X, [25, 75], method, 1, policy),
                                quantile(X, [.25, .75], 1, method, policy),
                                equal_nan=True)

    def test_extreme_finite_interpolation_is_safe(self):
        with np.errstate(over="raise", invalid="raise"):
            self.assertEqual(percentile([-1e308, 1e308], 50), 0)
            self.assertEqual(percentile([1e308, 1.6e308], 50, "midpoint"), 1.3e308)
            assert_allclose(percentile([-1e308, 1e308], [0, 25, 75, 100]),
                            [-1e308, -5e307, 5e307, 1e308])

    def test_non_contiguous_and_preserved_input(self):
        base = np.arange(40.0).reshape(5, 8)
        before = base.copy()
        view = base[::-1, ::2]
        result = percentile(view, [10, 90], axis=0)
        assert_allclose(result, np.percentile(view, [10, 90], axis=0))
        result[:] = -1
        assert_array_equal(base, before)

    def test_invalid_q_shapes_values_and_types(self):
        for q in (-1, 101, np.inf, np.nan, [[50]]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                percentile([1, 2], q)
        for q in (True, [True, 50], "50", 1j):
            with self.subTest(q=q), self.assertRaises(TypeError):
                percentile([1, 2], q)

    def test_invalid_data_axis_method_and_nan_policy(self):
        for values in ([np.inf], [-np.inf]):
            with self.assertRaises(ValueError):
                percentile(values, 50)
        for axis in (True, 1.0, (0,)):
            with self.assertRaises(TypeError):
                percentile([1], 50, axis=axis)
        with self.assertRaises(ValueError):
            percentile([1], 50, axis=1)
        for kwargs in ({"method": "nearest"}, {"nan_policy": "ignore"}):
            with self.assertRaises(ValueError):
                percentile([1], 50, **kwargs)
        for kwargs in ({"method": None}, {"nan_policy": 1}):
            with self.assertRaises(TypeError):
                percentile([1], 50, **kwargs)


class TestSortRankIntegration(unittest.TestCase):

    def test_ranking_topk_and_search_agree(self):
        scores = np.array([40, 10, 10, 20])
        values, indices = topk(scores, 2)
        assert_array_equal(indices, [0, 3])
        assert_array_equal(rank(scores)[indices], [4, 3])
        assert_array_equal(values, [40, 20])
        ordered = stable_sort(scores)
        self.assertEqual(binary_search(ordered, percentile(scores, 50)), (2, False))


if __name__ == "__main__":
    unittest.main()
