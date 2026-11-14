"""Independent batch references and lifecycle checks for streaming scaling."""

import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import StandardScaler


class StandardScalerTests(unittest.TestCase):
    def test_hand_calculated_population_statistics(self):
        scaler = StandardScaler().fit([[1, 10], [3, 10], [5, 10]])

        assert_array_equal(scaler.n_samples_seen_, [3, 3])
        assert_allclose(scaler.mean_, [3, 10])
        assert_allclose(scaler.var_, [8 / 3, 0])
        assert_allclose(scaler.scale_, [np.sqrt(8 / 3), 1])
        assert_allclose(scaler.transform([[3, 10]]), [[0, 0]])

    def test_unequal_chunks_match_independent_numpy_reference(self):
        X = np.random.default_rng(42).normal(size=(103, 5))
        X[::7, 1] = np.nan
        X[::3, 4] = np.nan

        scaler = StandardScaler()
        for chunk in np.split(X, [1, 4, 33, 101]):
            self.assertIs(scaler.partial_fit(chunk), scaler)

        assert_array_equal(scaler.n_samples_seen_, np.sum(~np.isnan(X), axis=0),)
        assert_allclose(scaler.mean_, np.nanmean(X, axis=0), atol=1e-14)
        assert_allclose(scaler.var_, np.nanvar(X, axis=0), rtol=1e-13)

        expected = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
        assert_allclose(scaler.transform(X), expected, atol=1e-13)

    def test_single_rows_and_reversed_chunk_order(self):
        X = np.arange(60, dtype=float).reshape(20, 3)
        scaler = StandardScaler()

        for row in X[::-1]:
            scaler.partial_fit(row[None, :])

        assert_allclose(scaler.mean_, X.mean(axis=0))
        assert_allclose(scaler.var_, X.var(axis=0))

    def test_large_offset_small_variance(self):
        X = 1e12 + np.arange(16, dtype=float)[:, None]
        scaler = StandardScaler().partial_fit(X[:4])
        scaler.partial_fit(X[4:])

        assert_allclose(scaler.mean_, X.mean(axis=0), rtol=0, atol=1e-4)
        assert_allclose(scaler.var_, X.var(axis=0), rtol=1e-12)

    def test_missing_column_initial_state_and_recovery(self):
        scaler = StandardScaler().fit([
            [np.nan, 1],
            [np.nan, 3],
        ])

        assert_array_equal(scaler.n_samples_seen_, [0, 2])
        assert_allclose(scaler.mean_, [np.nan, 2])
        assert_allclose(scaler.var_, [np.nan, 1])
        assert_allclose(scaler.scale_, [1, 1])
        assert_allclose(
            scaler.transform([[7, np.nan]]),
            [[np.nan, np.nan]],
        )

        scaler.partial_fit([
            [4, np.nan],
            [8, np.nan],
        ])

        assert_array_equal(scaler.n_samples_seen_, [2, 2])
        assert_allclose(scaler.mean_, [6, 2])
        assert_allclose(scaler.var_, [4, 1])

    def test_all_missing_update_preserves_statistics(self):
        scaler = StandardScaler().fit([[1, 2], [3, 4]])
        old = copy.deepcopy(scaler.__dict__)

        scaler.partial_fit(np.full((4, 2), np.nan))

        self.assert_state_equal(scaler.__dict__, old)

    def test_fit_resets_statistics_and_allows_new_schema(self):
        scaler = StandardScaler().fit([[1, 2], [3, 4]])

        self.assertIs(scaler.fit([[8], [10]]), scaler)
        self.assertEqual(scaler.n_features_in_, 1)
        assert_array_equal(scaler.n_samples_seen_, [2])
        assert_allclose(scaler.mean_, [9])

    def test_invalid_learning_preserves_state(self):
        scaler = StandardScaler().fit([[1, 2], [3, 4]])
        old = copy.deepcopy(scaler.__dict__)

        bad_inputs = [
            [],
            [1, 2],
            np.empty((0, 2)),
            np.empty((2, 0)),
            [[np.inf, 1]],
            [[True, False]],
            [["1", "2"]],
            np.array([[1, 2]], dtype=object),
            [[1j, 2j]],
        ]

        for method in (scaler.fit, scaler.partial_fit):
            for X in bad_inputs:
                with self.subTest(method=method.__name__, X=repr(X)):
                    with self.assertRaises((ValueError, TypeError)):
                        method(X)
                    self.assert_state_equal(scaler.__dict__, old)

        with self.assertRaises(ValueError):
            scaler.partial_fit([[1]])

        self.assert_state_equal(scaler.__dict__, old)

    def test_numeric_overflow_preserves_state(self):
        scaler = StandardScaler().fit([[1], [3]])
        old = copy.deepcopy(scaler.__dict__)

        for method in (scaler.fit, scaler.partial_fit):
            with self.subTest(method=method.__name__):
                with self.assertRaises(FloatingPointError):
                    method([[-1e308], [1e308]])
                self.assert_state_equal(scaler.__dict__, old)

    def test_input_and_output_ownership(self):
        X = np.array([[1., 4.], [3., 8.]])
        original = X.copy()

        scaler = StandardScaler().fit(X)
        output = scaler.transform(X)

        assert_array_equal(X, original)
        self.assertFalse(np.shares_memory(output, X))

        X[:] = -99
        output[:] = 99
        assert_allclose(scaler.mean_, [2, 6])

    def test_snapshot_remains_independent(self):
        scaler = StandardScaler().fit([[1], [3]])
        frozen = copy.deepcopy(scaler)
        expected = frozen.transform([[4]])

        scaler.partial_fit([[100]])

        assert_array_equal(frozen.transform([[4]]), expected)
        assert_array_equal(frozen.n_samples_seen_, [2])
        self.assertFalse(np.shares_memory(scaler.mean_, frozen.mean_))

    def test_read_only_noncontiguous_integer_input(self):
        X = np.arange(48).reshape(8, 6)[::2, ::2]
        X.flags.writeable = False

        scaler = StandardScaler().fit(X)

        assert_allclose(
            scaler.transform(X),
            (X - X.mean(axis=0)) / X.std(axis=0),
        )

    def test_transform_does_not_learn(self):
        scaler = StandardScaler().fit([[1], [3]])
        old = copy.deepcopy(scaler.__dict__)

        scaler.transform([[100], [np.nan]])

        self.assert_state_equal(scaler.__dict__, old)

    def test_transform_validation_and_empty_shape(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1]])

        scaler = StandardScaler().fit([[1, 2]])

        self.assertEqual(scaler.transform(np.empty((0, 2))).shape, (0, 2),)

        for X in (
            [[1]],
            [[np.inf, 1]],
            [1, 2],
            [["1", "2"]],
        ):
            with self.subTest(X=X):
                with self.assertRaises((ValueError, TypeError)):
                    scaler.transform(X)

    def test_fit_transform_accepts_optional_y(self):
        X = np.array([[1., 2.], [3., 4.]])

        assert_allclose(
            StandardScaler().fit_transform(X, [0, 1]),
            [[-1, -1], [1, 1]],
        )

    def test_training_does_not_retain_observations(self):
        scaler = StandardScaler()

        for _ in range(5):
            scaler.partial_fit(np.ones((100, 3)))

        arrays = [
            value
            for value in scaler.__dict__.values()
            if isinstance(value, np.ndarray)
        ]

        self.assertTrue(all(value.shape == (3,) for value in arrays))
        assert_array_equal(scaler.n_samples_seen_, [500, 500, 500])

    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())

        for key in expected:
            assert_array_equal(actual[key], expected[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()