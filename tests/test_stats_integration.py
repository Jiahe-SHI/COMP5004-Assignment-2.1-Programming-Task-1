"""Integration tests for statistics and preprocessing."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute.preprocessing import SimpleImputer, StandardScaler
from numcompute.stats import (
    mean,
    variance,
    median,
    quantile,
    histogram,
    StreamingStats,
)


class TestStatsIntegration(unittest.TestCase):

    def test_scaler_statistics_with_nan_and_constant_column(self):
        X = np.array([
            [1.0, 10.0, 7.0],
            [3.0, np.nan, 7.0],
            [5.0, 30.0, 7.0],
            [7.0, 50.0, 7.0],
        ])
        before = X.copy()

        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)

        # Both modules must use the same per-column NaN semantics.
        assert_allclose(mean(X, axis=0, nan_policy="omit"),[4.0, 30.0, 7.0],)
        assert_allclose(scaler.mean_,mean(X, axis=0, nan_policy="omit"),)

        # Nonconstant columns have population variance 1.
        # The constant column becomes zero.
        assert_allclose(mean(transformed, axis=0, nan_policy="omit"),[0.0, 0.0, 0.0],atol=1e-14,)
        assert_allclose(variance(transformed, axis=0, nan_policy="omit"),[1.0, 1.0, 0.0],atol=1e-14,)

        self.assertTrue(np.isnan(transformed[1, 1]))
        assert_array_equal(X, before)

    def test_imputation_changes_statistics_as_expected(self):
        X = np.array([[1.0], [np.nan], [3.0]])
        before = X.copy()

        filled = SimpleImputer(strategy="constant", fill_value=0.0,).fit_transform(X)

        # Ignoring a missing value differs from filling it with zero.
        self.assertEqual(mean(X, nan_policy="omit"), 2.0)
        self.assertAlmostEqual(mean(filled), 4.0 / 3.0)

        assert_allclose(quantile(filled, [0.0, 0.5, 1.0]),[0.0, 1.0, 3.0],)
        self.assertEqual(median(filled), 1.0)

        # Histogram requires a 1D input.
        counts, edges = histogram(filled[:, 0], bins=[0.0, 1.0, 3.0],)

        assert_array_equal(counts, [1, 2])
        assert_allclose(edges, [0.0, 1.0, 3.0])
        assert_array_equal(X, before)

    def test_streaming_matches_scaled_batch_statistics(self):
        X = np.array([
            [1.0, 10.0],
            [3.0, np.nan],
            [5.0, 30.0],
            [7.0, 50.0],
        ])

        transformed = StandardScaler().fit_transform(X)

        # StreamingStats handles one 1D stream at a time.
        for column in range(transformed.shape[1]):
            with self.subTest(column=column):
                values = transformed[:, column]
                stream = StreamingStats(nan_policy="omit")

                stream.update(values[:2])
                stream.update(values[2:])

                self.assertEqual(stream.count,int(np.count_nonzero(~np.isnan(values))),)
                assert_allclose(stream.mean, mean(values, nan_policy="omit"),atol=1e-14,)
                assert_allclose(stream.variance(ddof=1),variance(values, ddof=1, nan_policy="omit"), rtol=1e-12, atol=1e-14,)


if __name__ == "__main__":
    unittest.main()