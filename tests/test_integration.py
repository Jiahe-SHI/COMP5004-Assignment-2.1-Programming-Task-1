"""Integration checks using the actual demo CSV."""

import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from numcompute.io import load_csv, iter_csv_chunks
from numcompute.preprocessing import (
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)


class TestStageCIntegration(unittest.TestCase):

    def test_demo_data_processing(self):
        # Locate the demo file relative to this test file.
        path = (
            Path(__file__).resolve().parent.parent
            / "demo"
            / "sample_data.csv"
        )

        # Read age, hours and the constant column as numeric data.
        numeric = load_csv(
            path,
            skip_header=1,
            usecols=[0, 1, 3],
        )

        assert_allclose(
            numeric,
            [
                [20, 4, 1],
                [22, np.nan, 1],
                [np.nan, 6, 1],
                [24, 8, 1],
                [26, 10, 1],
            ],
            equal_nan=True,
        )

        # Chunked reading must match whole-file reading.
        chunks = list(iter_csv_chunks(
            path,
            chunk_size=2,
            skip_header=1,
            usecols=[0, 1, 3],
        ))

        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        assert_allclose(
            np.concatenate(chunks, axis=0),
            numeric,
            equal_nan=True,
        )

        # Fill numeric NaN values with zero.
        filled = SimpleImputer(fill_value=0).fit_transform(numeric)

        assert_array_equal(
            filled,
            [
                [20, 4, 1],
                [22, 0, 1],
                [0, 6, 1],
                [24, 8, 1],
                [26, 10, 1],
            ],
        )

        # Standardize numeric columns.
        standardized = StandardScaler().fit_transform(filled)

        assert_allclose(
            np.mean(standardized, axis=0),
            [0, 0, 0],
            atol=1e-12,
        )
        assert_allclose(
            np.std(standardized[:, :2], axis=0, ddof=0),
            [1, 1],
            atol=1e-12,
        )
        assert_array_equal(standardized[:, 2], np.zeros(5))

        # Check MinMaxScaler as an alternative numeric transformation.
        normalized = MinMaxScaler().fit_transform(filled)

        assert_allclose(
            normalized,
            [
                [20 / 26, 0.4, 0],
                [22 / 26, 0.0, 0],
                [0.0, 0.6, 0],
                [24 / 26, 0.8, 0],
                [1.0, 1.0, 0],
            ],
        )

        # Read the city column separately as text.
        cities = load_csv(
            path,
            skip_header=1,
            usecols=[2],
            dtype=str,
        )

        # CSV string missing values become empty strings.
        # Convert them to None for the encoder's missing-category rule.
        cities = cities.astype(object)
        cities[cities == ""] = None

        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(cities)

        assert_array_equal(
            encoder.categories_[0],
            np.array(
                ["Adelaide", "Melbourne", "Sydney", None],
                dtype=object,
            ),
        )
        assert_array_equal(
            encoded,
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        )

        # Combine standardized numeric data and encoded categories.
        combined = np.concatenate(
            [standardized, encoded],
            axis=1,
        )

        self.assertEqual(combined.shape, (5, 7))
        self.assertEqual(combined.dtype, np.dtype("float64"))
        self.assertTrue(np.all(np.isfinite(combined)))

        # Preprocessing must not overwrite the original missing values.
        self.assertTrue(np.isnan(numeric[1, 1]))
        self.assertTrue(np.isnan(numeric[2, 0]))


if __name__ == "__main__":
    unittest.main()