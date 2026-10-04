"""Tests for descriptive statistics."""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute.stats import (
    mean,
    variance,
    std,
    min as stats_min,
    max as stats_max,
    median,
    quantile,
    histogram,
    StreamingStats
)


class TestMean(unittest.TestCase):

    def test_mean_all_values(self):
        self.assertEqual(mean([[1, 2], [3, 4]]), 2.5)

    def test_axis_and_negative_axis(self):
        X = [[1, 2], [3, 4]]

        assert_allclose(mean(X, axis=0), [2.0, 3.0])
        assert_allclose(mean(X, axis=1), [1.5, 3.5])
        assert_allclose(mean(X, axis=-1), [1.5, 3.5])

    def test_three_dimensional_shape(self):
        X = np.arange(24).reshape(2, 3, 4)

        result = mean(X, axis=1)

        self.assertEqual(result.shape, (2, 4))
        assert_allclose(result, np.mean(X, axis=1))

    def test_nan_propagates_per_slice(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        assert_allclose(mean(X, axis=0),[2.0, np.nan], equal_nan=True,)

    def test_nan_omit(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        assert_allclose(mean(X, axis=0, nan_policy="omit"),[2.0, 4.0],)

    def test_nan_raise(self):
        with self.assertRaises(ValueError):
            mean([1.0, np.nan], nan_policy="raise")

    def test_empty_and_all_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            self.assertTrue(np.isnan(mean([])))

            self.assertTrue(np.isnan(mean([np.nan], nan_policy="omit")))

            assert_allclose(mean(np.empty((0, 3)), axis=0),[np.nan, np.nan, np.nan],equal_nan=True,)

            result = mean(np.empty((0, 3)), axis=1)
            self.assertEqual(result.shape, (0,))

    def test_scalar_and_output_dtype(self):
        result = mean(5)

        self.assertIsInstance(result, np.float64)
        self.assertEqual(result, 5.0)

        self.assertEqual(mean([[1, 2]], axis=0).dtype,np.dtype("float64"),)

    def test_non_contiguous_input_is_unchanged(self):
        original = np.arange(24.0).reshape(4, 6)
        before = original.copy()
        X = original[:, ::2]

        self.assertFalse(X.flags.c_contiguous)

        assert_allclose(mean(X, axis=0), np.mean(X, axis=0))
        assert_array_equal(original, before)

    def test_invalid_input_types(self):
        for X in ([True, False], [1 + 2j], ["1", "2"]):
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    mean(X)

        with self.assertRaises(TypeError):
            mean(np.array([1, 2], dtype=object))

    def test_infinity_is_rejected(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    mean([1.0, value])

    def test_invalid_axis(self):
        for axis in (True, 1.5, (0, 1)):
            with self.subTest(axis=axis):
                with self.assertRaises(TypeError):
                    mean([[1, 2]], axis=axis)

        with self.assertRaises(ValueError):
            mean([[1, 2]], axis=2)

        with self.assertRaises(ValueError):
            mean(5, axis=0)

    def test_invalid_nan_policy(self):
        with self.assertRaises(TypeError):
            mean([1, 2], nan_policy=None)

        with self.assertRaises(ValueError):
            mean([1, 2], nan_policy="ignore")

class TestVarianceAndStd(unittest.TestCase):

    def test_population_variance(self):
        # Mean = 2.5; squared deviations sum to 5.
        self.assertEqual(variance([1, 2, 3, 4]), 1.25)

    def test_sample_variance(self):
        self.assertAlmostEqual(variance([1, 2, 3, 4], ddof=1),5.0 / 3.0,)

    def test_standard_deviation(self):
        self.assertAlmostEqual(std([1, 2, 3, 4]),np.sqrt(1.25),)

        self.assertAlmostEqual(std([1, 2, 3, 4], ddof=1),np.sqrt(5.0 / 3.0),)

    def test_axis_and_negative_axis(self):
        X = [[1, 2], [3, 4]]

        assert_allclose(variance(X, axis=0), [1.0, 1.0])
        assert_allclose(variance(X, axis=1), [0.25, 0.25])
        assert_allclose(std(X, axis=-1), [0.5, 0.5])

    def test_three_dimensional_input(self):
        X = np.arange(24.0).reshape(2, 3, 4)

        for axis in (None, 0, 1, -1):
            with self.subTest(axis=axis):
                assert_allclose(variance(X, axis=axis, ddof=1),np.var(X, axis=axis, ddof=1),)
                assert_allclose(std(X, axis=axis, ddof=1),np.std(X, axis=axis, ddof=1),)

    def test_nan_propagates_per_slice(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        for function in (variance, std):
            with self.subTest(function=function.__name__):
                assert_allclose(function(X, axis=0),[1.0, np.nan],equal_nan=True,)

    def test_nan_omit_uses_per_slice_count(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        assert_allclose(variance(X, axis=0, nan_policy="omit"),[1.0, 0.0],)

        assert_allclose(variance(X, axis=0, ddof=1, nan_policy="omit"),[2.0, np.nan],equal_nan=True,)

        assert_allclose(std(X, axis=0, ddof=1, nan_policy="omit"),[np.sqrt(2.0), np.nan],equal_nan=True,)

    def test_nan_raise(self):
        for function in (variance, std):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function([1.0, np.nan], nan_policy="raise")

    def test_empty_all_nan_and_ddof_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            for function in (variance, std):
                with self.subTest(function=function.__name__):
                    self.assertTrue(np.isnan(function([])))

                    self.assertTrue(np.isnan(function([np.nan], nan_policy="omit")))

                    self.assertTrue(np.isnan(function([1, 2], ddof=2)))

                    self.assertTrue(np.isnan(function([1, 2], ddof=3)))

                    assert_allclose(function(np.empty((0, 3)), axis=0),[np.nan, np.nan, np.nan],equal_nan=True,)

                    self.assertEqual(function(np.empty((0, 3)), axis=1).shape,(0,),)

    def test_constant_and_single_value(self):
        for function in (variance, std):
            with self.subTest(function=function.__name__):
                self.assertEqual(function([7, 7, 7]), 0.0)
                self.assertEqual(function([7]), 0.0)
                self.assertTrue(np.isnan(function([7], ddof=1)))

    def test_large_mean_small_variance(self):
        # These offsets remain exactly representable at this scale.
        X = 1e12 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        self.assertAlmostEqual(variance(X), 2.0)
        self.assertAlmostEqual(variance(X, ddof=1), 2.5)
        self.assertAlmostEqual(std(X), np.sqrt(2.0))

    def test_scalar_and_output_dtype(self):
        for function in (variance, std):
            with self.subTest(function=function.__name__):
                result = function(5)

                self.assertIsInstance(result, np.float64)
                self.assertEqual(result, 0.0)

                self.assertEqual(function([[1, 2], [3, 4]], axis=0).dtype, np.dtype("float64"),)

    def test_non_contiguous_input_is_unchanged(self):
        original = np.arange(24.0).reshape(4, 6)
        before = original.copy()
        X = original[:, ::2]

        assert_allclose(variance(X, axis=0),np.var(X, axis=0),)
        assert_allclose(std(X, axis=0), np.std(X, axis=0),)
        assert_array_equal(original, before)

    def test_invalid_ddof(self):
        for function in (variance, std):
            for ddof in (True, 1.5, "1", None):
                with self.subTest(function=function.__name__,ddof=ddof,):
                    with self.assertRaises(TypeError):
                        function([1, 2, 3], ddof=ddof)

            with self.assertRaises(ValueError):
                function([1, 2, 3], ddof=-1)

    def test_numpy_integer_ddof(self):
        self.assertEqual(variance([1, 3], ddof=np.int64(1)),2.0,)

    def test_infinity_is_rejected(self):
        for function in (variance, std):
            for value in (np.inf, -np.inf):
                with self.subTest(function=function.__name__, value=value,):
                    with self.assertRaises(ValueError):
                        function([1.0, value])

class TestMinMaxMedian(unittest.TestCase):

    def test_min_max(self):
        X = [3, -2, 7, 1]

        self.assertEqual(stats_min(X), -2.0)
        self.assertEqual(stats_max(X), 7.0)

    def test_median_odd_even_and_duplicates(self):
        self.assertEqual(median([9, 1, 3]), 3.0)
        self.assertEqual(median([8, 2, 4, 6]), 5.0)
        self.assertEqual(median([2, 2, 2, 9]), 2.0)

    def test_axis(self):
        X = [[1, 8, 3], [5, 2, 7]]

        assert_allclose(stats_min(X, axis=0), [1, 2, 3])
        assert_allclose(stats_max(X, axis=0), [5, 8, 7])
        assert_allclose(median(X, axis=0), [3, 5, 5])

        assert_allclose(stats_min(X, axis=1), [1, 2])
        assert_allclose(stats_max(X, axis=1), [8, 7])
        assert_allclose(median(X, axis=1), [3, 5])

    def test_three_dimensions_and_negative_axes(self):
        X = np.arange(24.0).reshape(2, 3, 4)

        pairs = (
            (stats_min, np.min),
            (stats_max, np.max),
            (median, np.median),
        )

        for function, reference in pairs:
            for axis in (None, 0, 1, -1, -2):
                with self.subTest(function=function.__name__, axis=axis,):
                    assert_allclose(function(X, axis=axis),reference(X, axis=axis),)

    def test_nan_propagates_per_slice(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        cases = (
            (stats_min, [1.0, np.nan]),
            (stats_max, [3.0, np.nan]),
            (median, [2.0, np.nan]),
        )

        for function, expected in cases:
            with self.subTest(function=function.__name__):
                assert_allclose(function(X, axis=0),expected,equal_nan=True,)

    def test_nan_omit(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        cases = (
            (stats_min, [1.0, 4.0]),
            (stats_max, [3.0, 4.0]),
            (median, [2.0, 4.0]),
        )

        for function, expected in cases:
            with self.subTest(function=function.__name__):
                assert_allclose(function(X, axis=0, nan_policy="omit"),expected,)

    def test_median_different_valid_counts(self):
        X = [
            [9.0, 1.0, 3.0, np.nan],
            [8.0, 2.0, 4.0, 6.0],
            [np.nan, np.nan, 7.0, np.nan],
        ]

        assert_allclose(median(X, axis=1, nan_policy="omit"),[3.0, 5.0, 7.0],)

    def test_nan_raise(self):
        for function in (stats_min, stats_max, median):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function([1.0, np.nan], nan_policy="raise")

    def test_empty_and_all_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            for function in (stats_min, stats_max, median):
                with self.subTest(function=function.__name__):
                    self.assertTrue(np.isnan(function([])))

                    self.assertTrue(np.isnan(function([np.nan], nan_policy="omit")))

                    assert_allclose(function(np.empty((0, 3)), axis=0),[np.nan, np.nan, np.nan],equal_nan=True,)

                    self.assertEqual(function(np.empty((0, 3)), axis=1).shape,(0,),)

                    assert_allclose(function([[np.nan, 1.0], [np.nan, 1.0]],axis=0, 
                                             nan_policy="omit",),[np.nan, 1.0],equal_nan=True,)

    def test_scalar_and_dtype(self):
        for function in (stats_min, stats_max, median):
            with self.subTest(function=function.__name__):
                result = function(5)

                self.assertIsInstance(result, np.float64)
                self.assertEqual(result, 5.0)

                self.assertEqual(function([[1, 2]], axis=0).dtype,np.dtype("float64"),)

    def test_non_contiguous_input_is_unchanged(self):
        original = np.arange(24.0).reshape(4, 6)
        before = original.copy()
        X = original[:, ::2]

        for function, reference in (
            (stats_min, np.min),
            (stats_max, np.max),
            (median, np.median),
        ):
            with self.subTest(function=function.__name__):
                assert_allclose(function(X, axis=0),reference(X, axis=0),)

        assert_array_equal(original, before)

    def test_large_finite_median_without_overflow(self):
        limit = np.finfo(np.float64).max

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            self.assertEqual(median([limit, limit]), limit)
            self.assertEqual(median([-limit, -limit]), -limit)
            self.assertEqual(median([-limit, limit]), 0.0)

    def test_small_equal_values_preserved(self):
        tiny = np.nextafter(0.0, 1.0)

        self.assertEqual(median([tiny, tiny]), tiny)
        self.assertEqual(median([-tiny, -tiny]), -tiny)

    def test_invalid_inputs_and_parameters(self):
        for function in (stats_min, stats_max, median):
            with self.subTest(function=function.__name__):
                with self.assertRaises(TypeError):
                    function(["1", "2"])

                with self.assertRaises(TypeError):
                    function([1, 2], axis=True)

                with self.assertRaises(ValueError):
                    function([1, 2], axis=1)

                with self.assertRaises(ValueError):
                    function([1, np.inf])

                with self.assertRaises(ValueError):
                    function([1, -np.inf])

                with self.assertRaises(ValueError):
                    function([1, 2], nan_policy="ignore")

class TestQuantile(unittest.TestCase):

    def test_endpoints_and_median(self):
        assert_allclose(quantile([40, 10, 30, 20], [0, 0.5, 1]),[10, 25, 40],)

    def test_four_interpolation_methods(self):
        # Position = 0.25 * (4 - 1) = 0.75.
        X = [0, 10, 20, 30]

        expected = {
            "linear": 7.5,
            "lower": 0.0,
            "higher": 10.0,
            "midpoint": 5.0,
        }

        for method, value in expected.items():
            with self.subTest(method=method):
                self.assertEqual(quantile(X, 0.25, method=method),value,)

    def test_exact_position_for_all_methods(self):
        for method in ("linear", "lower", "higher", "midpoint"):
            with self.subTest(method=method):
                self.assertEqual(quantile([10, 20, 30], 0.5, method=method),20.0,)

    def test_scalar_q_and_vector_q_shapes(self):
        X = [[1, 10], [3, 30], [5, 50]]

        scalar_result = quantile(X, 0.5, axis=0)
        vector_result = quantile(X, [0.5], axis=0)

        self.assertEqual(scalar_result.shape, (2,))
        self.assertEqual(vector_result.shape, (1, 2))

        assert_allclose(scalar_result, [3, 30])
        assert_allclose(vector_result, [[3, 30]])

    def test_multiple_quantiles_and_axes(self):
        X = np.arange(24.0).reshape(2, 3, 4)
        q = [0.0, 0.2, 0.5, 0.8, 1.0]

        for axis in (None, 0, 1, -1):
            for method in ("linear", "lower", "higher", "midpoint"):
                with self.subTest(axis=axis, method=method):
                    assert_allclose(quantile(X, q, axis=axis, method=method),
                                    np.quantile(X, q, axis=axis, method=method),)

    def test_q_order_and_duplicates_preserved(self):
        assert_allclose(quantile([0, 10, 20], [1, 0, 0.5, 0.5]),[20, 0, 10, 10],)

    def test_nan_propagates_per_slice(self):
        X = [[1.0, np.nan], [3.0, 4.0]]

        assert_allclose(quantile(X, [0, 0.5, 1], axis=0), [[1, np.nan], [2, np.nan], [3, np.nan]],equal_nan=True,)

    def test_nan_omit_uses_per_slice_count(self):
        X = [
            [0.0, 10.0, 20.0, np.nan],
            [0.0, 10.0, 20.0, 30.0],
        ]

        assert_allclose(quantile(X, [0.25, 0.5], axis=1, nan_policy="omit"),[[5.0, 7.5], [10.0, 15.0]],)

    def test_nan_raise(self):
        with self.assertRaises(ValueError):
            quantile([1.0, np.nan], 0.5, nan_policy="raise")

    def test_empty_and_all_nan_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            self.assertTrue(np.isnan(quantile([], 0.5)))

            assert_allclose(quantile([np.nan], [0, 0.5, 1], nan_policy="omit"),[np.nan, np.nan, np.nan], equal_nan=True,)

            result = quantile(np.empty((0, 3)), [0.25, 0.75], axis=0)
            self.assertEqual(result.shape, (2, 3))
            self.assertTrue(np.all(np.isnan(result)))

            result = quantile(np.empty((0, 3)), [0.25, 0.75], axis=1)
            self.assertEqual(result.shape, (2, 0))

    def test_empty_q(self):
        self.assertEqual(quantile([1, 2, 3], []).shape, (0,))
        self.assertEqual(quantile([], []).shape, (0,))

        result = quantile([[1, 2], [3, 4]], [], axis=0)
        self.assertEqual(result.shape, (0, 2))

    def test_scalar_input_and_return_dtype(self):
        result = quantile(7, 0.25)

        self.assertIsInstance(result, np.float64)
        self.assertEqual(result, 7.0)

        assert_allclose(quantile(7, [0, 0.5, 1]), [7, 7, 7])

    def test_large_finite_interpolation(self):
        limit = np.finfo(np.float64).max

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            assert_allclose(quantile([-limit, limit], [0, 0.25, 0.5, 0.75, 1]),
                            [-limit, -limit / 2, 0, limit / 2, limit],)

            self.assertEqual(quantile([limit, limit], 0.5),limit,)

            self.assertEqual(quantile([-limit, limit], 0.5, method="midpoint"),0.0,)

    def test_non_contiguous_input_and_q_unchanged(self):
        original = np.arange(24.0).reshape(4, 6)
        before = original.copy()
        X = original[:, ::2]

        q = np.array([0.75, 0.25])
        q_before = q.copy()

        assert_allclose(quantile(X, q, axis=0),np.quantile(X, q, axis=0),)

        assert_array_equal(original, before)
        assert_array_equal(q, q_before)

    def test_invalid_q(self):
        for q in (-0.1, 1.1, np.nan, np.inf, [[0.5]]):
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    quantile([1, 2, 3], q)

        for q in (True, "0.5", [1 + 2j]):
            with self.subTest(q=q):
                with self.assertRaises(TypeError):
                    quantile([1, 2, 3], q)

    def test_invalid_method_and_input(self):
        with self.assertRaises(TypeError):
            quantile([1, 2], 0.5, method=None)

        with self.assertRaises(ValueError):
            quantile([1, 2], 0.5, method="nearest")

        with self.assertRaises(ValueError):
            quantile([1, np.inf], 0.5)

        with self.assertRaises(ValueError):
            quantile([1, 2], 0.5, axis=1)

        with self.assertRaises(TypeError):
            quantile([1, 2], 0.5, axis=True)

class TestHistogram(unittest.TestCase):

    def test_explicit_edges_and_boundary_rules(self):
        counts, edges = histogram([0, 1, 2, 3, 4], bins=[0, 2, 4],)

        assert_array_equal(counts, [2, 3])
        assert_allclose(edges, [0, 2, 4])

    def test_integer_bins(self):
        counts, edges = histogram([0, 1, 2, 3, 4],bins=2,)

        assert_array_equal(counts, [2, 3])
        assert_allclose(edges, [0, 2, 4])

    def test_explicit_range_ignores_outside_values(self):
        counts, edges = histogram([-1, 0, 1, 2, 3, 4, 5],bins=2,range=(0, 4),)

        assert_array_equal(counts, [2, 3])
        assert_allclose(edges, [0, 2, 4])

    def test_nonuniform_edges(self):
        counts, edges = histogram([-1, 0, 0.5, 1, 2, 3, 10, 11], bins=[0, 1, 3, 10],)

        assert_array_equal(counts, [2, 2, 2])
        assert_allclose(edges, [0, 1, 3, 10])

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=100)

        # Use identical explicit edges to isolate counting behavior.
        edges = np.linspace(-2, 2, 9)

        counts, actual_edges = histogram(X, bins=edges)
        expected_counts, expected_edges = np.histogram(X, bins=edges)

        assert_array_equal(counts, expected_counts)
        assert_array_equal(actual_edges, expected_edges)

    def test_nan_omitted_by_default(self):
        counts, edges = histogram([0, np.nan, 1, 2],bins=[0, 1, 2],)

        assert_array_equal(counts, [1, 2])
        assert_allclose(edges, [0, 1, 2])

    def test_nan_raise(self):
        with self.assertRaises(ValueError):
            histogram([1, np.nan], nan_policy="raise")

    def test_invalid_nan_policy(self):
        for policy in ("propagate", "ignore"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    histogram([1, 2], nan_policy=policy)

        with self.assertRaises(TypeError):
            histogram([1, 2], nan_policy=None)

    def test_empty_and_all_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            for X in ([], [np.nan, np.nan]):
                with self.subTest(X=X):
                    counts, edges = histogram(X, bins=2)

                    assert_array_equal(counts, [0, 0])
                    assert_allclose(edges, [0, 0.5, 1])

    def test_empty_with_range_or_edges(self):
        counts, edges = histogram([], bins=2, range=(-2, 2))

        assert_array_equal(counts, [0, 0])
        assert_allclose(edges, [-2, 0, 2])

        counts, edges = histogram([], bins=[1, 3, 9])

        assert_array_equal(counts, [0, 0])
        assert_allclose(edges, [1, 3, 9])

    def test_constant_data(self):
        counts, edges = histogram([5, 5, 5], bins=2)

        assert_array_equal(counts, [0, 3])
        assert_allclose(edges, [4.5, 5.0, 5.5])

    def test_output_types(self):
        counts, edges = histogram([1, 2, 3], bins=np.int64(2))

        self.assertTrue(np.issubdtype(counts.dtype, np.integer))
        self.assertEqual(edges.dtype, np.dtype("float64"))

    def test_invalid_integer_bins(self):
        for bins in (0, -1):
            with self.subTest(bins=bins):
                with self.assertRaises(ValueError):
                    histogram([1, 2], bins=bins)

        for bins in (True, np.bool_(False), 2.5, "2"):
            with self.subTest(bins=bins):
                with self.assertRaises(TypeError):
                    histogram([1, 2], bins=bins)

    def test_invalid_edge_arrays(self):
        for edges in (
            [],
            [0],
            [0, 0, 1],
            [2, 1, 0],
            [0, np.nan, 2],
            [0, np.inf],
            [[0, 1], [2, 3]],
        ):
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError):
                    histogram([1, 2], bins=edges)

    def test_invalid_range(self):
        for bounds in (
            (1, 1),
            (2, 1),
            (0, np.inf),
            (0, np.nan),
            (0,),
            (0, 1, 2),
        ):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    histogram([1, 2], bins=2, range=bounds)

        with self.assertRaises(ValueError):
            histogram(
                [1, 2],
                bins=[0, 1, 2],
                range=(0, 2),
            )

    def test_invalid_values(self):
        for X in (1, [[1, 2]], [1, np.inf], [1, -np.inf]):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    histogram(X)

        for X in ([True, False], ["1", "2"], [1 + 2j]):
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    histogram(X)

    def test_extreme_finite_edges(self):
        limit = np.finfo(np.float64).max

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            counts, edges = histogram([-limit, 0, limit], bins=2,range=(-limit, limit),)

            assert_array_equal(counts, [1, 2])
            assert_array_equal(edges, [-limit, 0, limit])

    def test_unrepresentable_edges_rejected(self):
        limit = np.finfo(np.float64).max

        # Adding or subtracting 0.5 cannot expand this value.
        with self.assertRaises(ValueError):
            histogram([limit, limit], bins=2)

        # No float64 value exists strictly between these bounds.
        upper = np.nextafter(1.0, 2.0)

        with self.assertRaises(ValueError):
            histogram([], bins=2, range=(1.0, upper))

    def test_inputs_unchanged_and_edges_independent(self):
        original = np.arange(10.0)
        before = original.copy()
        X = original[::2]

        edges = np.array([0.0, 4.0, 8.0])
        edges_before = edges.copy()

        counts, returned_edges = histogram(X, bins=edges)

        assert_array_equal(counts, [2, 3])
        assert_array_equal(original, before)
        assert_array_equal(edges, edges_before)

        returned_edges[0] = -100.0
        assert_array_equal(edges, edges_before)

class TestStreamingStats(unittest.TestCase):

    def test_initial_state(self):
        stats = StreamingStats()

        self.assertEqual(stats.count, 0)
        self.assertTrue(np.isnan(stats.mean))
        self.assertEqual(stats.M2, 0.0)
        self.assertTrue(np.isnan(stats.variance()))
        self.assertTrue(np.isnan(stats.std()))

    def test_single_batch(self):
        stats = StreamingStats().update([1, 2, 3, 4])

        self.assertEqual(stats.count, 4)
        self.assertEqual(stats.mean, 2.5)
        self.assertEqual(stats.M2, 5.0)
        self.assertEqual(stats.variance(), 1.25)
        self.assertAlmostEqual(stats.std(), np.sqrt(1.25))

    def test_multiple_batches(self):
        stats = StreamingStats()
        stats.update([1, 2])
        stats.update([3, 4])

        self.assertEqual(stats.count, 4)
        self.assertEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.M2, 5.0)
        self.assertAlmostEqual(stats.variance(), 1.25)

    def test_update_returns_self(self):
        stats = StreamingStats()

        self.assertIs(stats.update([1, 2]), stats)
        self.assertIs(stats.update([3]), stats)

    def test_sample_variance(self):
        stats = StreamingStats().update([1, 2, 3, 4])

        self.assertAlmostEqual(stats.variance(ddof=1), 5.0 / 3.0)
        self.assertAlmostEqual(stats.std(ddof=1), np.sqrt(5.0 / 3.0),)

    def test_different_batch_sizes_match_numpy(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=103)

        for batch_size in (1, 7, 32, 103):
            with self.subTest(batch_size=batch_size):
                stats = StreamingStats()

                for start in range(0, X.size, batch_size):
                    stats.update(X[start:start + batch_size])

                self.assertEqual(stats.count, X.size)
                assert_allclose(stats.mean, np.mean(X), atol=1e-14)
                assert_allclose(stats.variance(ddof=1),np.var(X, ddof=1),rtol=1e-12,)
                assert_allclose(stats.std(ddof=1),np.std(X, ddof=1),rtol=1e-12,)

    def test_large_mean_small_variance(self):
        X = 1e12 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        stats = StreamingStats()
        stats.update(X[:2])
        stats.update(X[2:])

        self.assertEqual(stats.mean, 1e12)
        self.assertAlmostEqual(stats.variance(), 2.0)
        self.assertAlmostEqual(stats.variance(ddof=1), 2.5)

    def test_nan_omit(self):
        stats = StreamingStats(nan_policy="omit")
        stats.update([1, np.nan, 2])
        stats.update([np.nan, 3, 4])

        self.assertEqual(stats.count, 4)
        self.assertEqual(stats.mean, 2.5)
        self.assertAlmostEqual(stats.variance(), 1.25)

    def test_all_nan_omit_leaves_state_unchanged(self):
        stats = StreamingStats(nan_policy="omit")
        stats.update([np.nan])

        self.assertEqual(stats.count, 0)
        self.assertTrue(np.isnan(stats.mean))
        self.assertEqual(stats.M2, 0.0)

        stats.update([1, 3])
        before = (stats.count, stats.mean, stats.M2)

        stats.update([np.nan, np.nan])

        self.assertEqual((stats.count, stats.mean, stats.M2),before,)

    def test_nan_raise_preserves_state(self):
        stats = StreamingStats(nan_policy="raise")
        stats.update([1, 3])
        before = (stats.count, stats.mean, stats.M2)

        with self.assertRaises(ValueError):
            stats.update([5, np.nan, 7])

        self.assertEqual((stats.count, stats.mean, stats.M2),before,)

    def test_nan_propagates_until_reset(self):
        stats = StreamingStats()
        stats.update([1, np.nan, 2])

        self.assertEqual(stats.count, 2)
        self.assertTrue(np.isnan(stats.mean))
        self.assertTrue(np.isnan(stats.M2))
        self.assertTrue(np.isnan(stats.variance()))
        self.assertTrue(np.isnan(stats.std()))

        stats.update([3, 4])

        self.assertEqual(stats.count, 4)
        self.assertTrue(np.isnan(stats.mean))
        self.assertTrue(np.isnan(stats.M2))

    def test_all_nan_propagates_with_zero_count(self):
        stats = StreamingStats().update([np.nan, np.nan])

        self.assertEqual(stats.count, 0)
        self.assertTrue(np.isnan(stats.mean))
        self.assertTrue(np.isnan(stats.M2))

    def test_empty_update_leaves_state_unchanged(self):
        stats = StreamingStats()

        self.assertIs(stats.update([]), stats)
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.M2, 0.0)

        stats.update([1, 3])
        before = (stats.count, stats.mean, stats.M2)

        stats.update([])

        self.assertEqual((stats.count, stats.mean, stats.M2),before,)

    def test_reset_and_reuse(self):
        stats = StreamingStats().update([1, np.nan])

        self.assertIs(stats.reset(), stats)
        self.assertEqual(stats.count, 0)
        self.assertTrue(np.isnan(stats.mean))
        self.assertEqual(stats.M2, 0.0)
        self.assertEqual(stats.nan_policy, "propagate")

        stats.update([2, 4])

        self.assertEqual(stats.count, 2)
        self.assertEqual(stats.mean, 3.0)
        self.assertEqual(stats.variance(), 1.0)

    def test_constant_data_and_insufficient_count(self):
        stats = StreamingStats().update([7])

        self.assertEqual(stats.variance(), 0.0)
        self.assertTrue(np.isnan(stats.variance(ddof=1)))
        self.assertTrue(np.isnan(stats.std(ddof=2)))

        stats.update([7, 7])

        self.assertEqual(stats.M2, 0.0)
        self.assertEqual(stats.variance(ddof=1), 0.0)

    def test_invalid_ddof(self):
        stats = StreamingStats().update([1, 2, 3])

        for method in (stats.variance, stats.std):
            for ddof in (True, 1.5, "1"):
                with self.subTest(method=method.__name__,ddof=ddof,):
                    with self.assertRaises(TypeError):
                        method(ddof=ddof)

            with self.assertRaises(ValueError):
                method(ddof=-1)

        self.assertAlmostEqual(stats.variance(ddof=np.int64(1)),1.0,)

    def test_invalid_batch_preserves_state(self):
        stats = StreamingStats().update([1, 3])
        before = (stats.count, stats.mean, stats.M2)

        for X in ([5, np.inf], [5, -np.inf], [[5, 7]], 5):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    stats.update(X)

                self.assertEqual((stats.count, stats.mean, stats.M2),before,)

        for X in ([True, False], ["5", "7"], [1 + 2j]):
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    stats.update(X)

                self.assertEqual((stats.count, stats.mean, stats.M2),before,)

    def test_overflow_preserves_state(self):
        stats = StreamingStats().update([1, 3])
        before = (stats.count, stats.mean, stats.M2)

        limit = np.finfo(np.float64).max

        with self.assertRaises(ValueError):
            stats.update([-limit, limit])

        self.assertEqual((stats.count, stats.mean, stats.M2),before,)

    def test_non_contiguous_input_is_unchanged(self):
        original = np.arange(10.0)
        before = original.copy()
        X = original[::2]

        stats = StreamingStats().update(X)

        assert_allclose(stats.mean, np.mean(X))
        assert_allclose(stats.variance(), np.var(X))
        assert_array_equal(original, before)

    def test_invalid_nan_policy(self):
        with self.assertRaises(TypeError):
            StreamingStats(nan_policy=None)

        with self.assertRaises(ValueError):
            StreamingStats(nan_policy="ignore")

if __name__ == "__main__":
    unittest.main()