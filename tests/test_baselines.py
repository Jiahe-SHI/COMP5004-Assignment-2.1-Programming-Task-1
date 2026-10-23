import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from benchmark.baselines import loop_mean, loop_standardize
from numcompute.stats import mean
from numcompute.preprocessing import StandardScaler

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

if __name__ == "__main__":
    unittest.main()