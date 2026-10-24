import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from benchmark.baselines import (
    loop_mean, loop_standardize, loop_mse, full_sort_topk,
)
from numcompute.stats import mean
from numcompute.preprocessing import StandardScaler
from numcompute.metrics import mse
from numcompute.sort_search import topk

class TestLoopMean(unittest.TestCase):
    def test_known_result(self):
        self.assertEqual(loop_mean([1, 2, 3, 4]), 2.5)
        self.assertEqual(loop_mean([7, 7, 7]), 7.0)
    """
    Use a fixed seed and a non-contiguous array.
    """
    def test_matches_project_mean(self):
        data = np.random.default_rng(42).normal(size=1000)[::2]
        before = data.copy()

        assert_allclose(
            loop_mean(data), mean(data),
            rtol=1e-10, atol=1e-12,
        )
        # Ensure the function does not modify its input.
        assert_array_equal(data, before)

    def test_invalid_input(self):
        for data in ([], [[1, 2]], [np.nan], [np.inf]):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    loop_mean(data)

        with self.assertRaises(TypeError):
            loop_mean([True, 2])

class TestLoopStandardize(unittest.TestCase):
    def test_known_result_and_constant_column(self):
        result = loop_standardize([[1, 7], [3, 7]])
        assert_allclose(result, [[-1, 0], [1, 0]])

    def test_matches_project_scaler(self):
        # Test non-contiguous data with a constant column.
        data = np.random.default_rng(42).normal(size=(20, 6))[:, ::2]
        data[:, 1] = 7.0
        before = data.copy()

        result = loop_standardize(data)
        expected = StandardScaler().fit_transform(data)

        assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
        assert_array_equal(data, before)
        self.assertFalse(np.shares_memory(result, data))

    def test_invalid_input(self):
        cases = (
            [1, 2],
            np.empty((0, 2)),
            np.empty((2, 0)),
            [[np.nan]],
            [[np.inf]],
        )
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    loop_standardize(data)
class TestLoopMSE(unittest.TestCase):
    def test_known_results(self):
        # Check 1D results and the total-element denominator for 2D.
        self.assertAlmostEqual(
            loop_mse([1, 2, 3], [1, 4, 2]), 5 / 3
        )
        self.assertEqual(
            loop_mse([[1, 2], [3, 4]], [[1, 0], [5, 4]]),
            2.0,
        )

    def test_matches_project_mse(self):
        rng = np.random.default_rng(42)
        true = rng.normal(size=1000)[::2]
        pred = rng.normal(size=1000)[::2]
        true_before = true.copy()
        pred_before = pred.copy()

        assert_allclose(
            loop_mse(true, pred), mse(true, pred),
            rtol=1e-10, atol=1e-12,
        )
        assert_array_equal(true, true_before)
        assert_array_equal(pred, pred_before)

    def test_invalid_input(self):
        cases = (
            ([], []),
            ([1, 2], [1]),
            ([1, 2], [[1, 2]]),
            ([np.nan], [0]),
            ([0], [np.inf]),
            (1, 1),
        )
        for true, pred in cases:
            with self.subTest(true=true, pred=pred):
                with self.assertRaises(ValueError):
                    loop_mse(true, pred)

        with self.assertRaises(TypeError):
            loop_mse([True, 2], [1, 2])

class TestFullSortTopK(unittest.TestCase):
    def test_boundary_ties(self):
        # Earlier indices win when a tie crosses the cutoff.
        values, indices = full_sort_topk([5, 9, 9, 9, 4], 2)
        assert_array_equal(values, [9, 9])
        assert_array_equal(indices, [1, 2])

    def test_zero_and_full_selection(self):
        values, indices = full_sort_topk([], 0)
        self.assertEqual(values.shape, (0,))
        self.assertEqual(indices.shape, (0,))

        values, indices = full_sort_topk([2, 1, 2], 3)
        assert_array_equal(values, [2, 2, 1])
        assert_array_equal(indices, [0, 2, 1])

    def test_matches_project_topk(self):
        # Float64 data contains ties and uses a non-contiguous view.
        rng = np.random.default_rng(42)
        data = rng.integers(-5, 6, size=100).astype(np.float64)[::2]
        before = data.copy()

        for k in (0, 1, 10, data.size):
            with self.subTest(k=k):
                values, indices = full_sort_topk(data, k)
                expected_values, expected_indices = topk(data, k)
                assert_array_equal(values, expected_values)
                assert_array_equal(indices, expected_indices)

        assert_array_equal(data, before)

    def test_invalid_input(self):
        for data, k in (
            ([1, 2], -1),
            ([1, 2], 3),
            ([[1, 2]], 1),
            ([np.nan], 1),
            ([np.inf], 1),
        ):
            with self.subTest(data=data, k=k):
                with self.assertRaises(ValueError):
                    full_sort_topk(data, k)

        for k in (True, 1.5):
            with self.subTest(k=k):
                with self.assertRaises(TypeError):
                    full_sort_topk([1, 2], k)

if __name__ == "__main__":
    unittest.main()