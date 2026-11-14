"""Streaming numeric imputation: references, boundaries and lifecycle."""

import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import Imputer


class ImputerTests(unittest.TestCase):
    def assert_learned_equal(self, actual, expected):
        self.assertEqual(actual.n_features_in_, expected.n_features_in_)
        self.assertEqual(actual._fitted_config, expected._fitted_config)
        assert_array_equal(actual.n_samples_seen_, expected.n_samples_seen_)
        assert_array_equal(actual.statistics_, expected.statistics_)

        if expected._moments is None:
            self.assertIsNone(actual._moments)
        else:
            for current, saved in zip(actual._moments, expected._moments):
                assert_array_equal(current, saved)

    def test_hand_calculated_mean_imputation(self):
        imputer = Imputer().fit([
            [1.0, np.nan],
            [3.0, 10.0],
            [8.0, 14.0],
        ])

        assert_array_equal(imputer.n_samples_seen_, [3, 2])
        assert_allclose(imputer.statistics_, [4.0, 12.0])
        assert_allclose(
            imputer.transform([[np.nan, np.nan], [5.0, 7.0]]),
            [[4.0, 12.0], [5.0, 7.0]],
        )

    def test_unequal_chunks_match_numpy_reference(self):
        X = np.random.default_rng(17).normal(size=(101, 4))
        X[::3, 0] = np.nan
        X[::5, 2] = np.nan

        imputer = Imputer()
        for chunk in np.split(X, [1, 7, 40, 99]):
            self.assertIs(imputer.partial_fit(chunk), imputer)

        expected_mean = np.nanmean(X, axis=0)
        expected_output = np.where(np.isnan(X), expected_mean, X)

        assert_array_equal(imputer.n_samples_seen_, np.sum(~np.isnan(X), axis=0),)
        assert_allclose(imputer.statistics_, expected_mean, atol=1e-14)
        assert_allclose(imputer.transform(X), expected_output, atol=1e-14)

    def test_all_missing_default_fallback(self):
        imputer = Imputer().fit(np.full((3, 2), np.nan))

        assert_array_equal(imputer.n_samples_seen_, [0, 0])
        assert_allclose(imputer.statistics_, [0.0, 0.0])
        assert_allclose(
            imputer.transform([[np.nan, 5.0]]),
            [[0.0, 5.0]],
        )

    def test_custom_fallback_and_later_recovery(self):
        imputer = Imputer(fill_value=-9).partial_fit([
            [1.0, np.nan],
            [3.0, np.nan],
        ])

        assert_allclose(imputer.statistics_, [2.0, -9.0])

        imputer.partial_fit([
            [np.nan, 10.0],
            [np.nan, 14.0],
        ])

        assert_array_equal(imputer.n_samples_seen_, [2, 2])
        assert_allclose(imputer.statistics_, [2.0, 12.0])

    def test_all_missing_update_preserves_estimates(self):
        for strategy in ("mean", "constant"):
            with self.subTest(strategy=strategy):
                imputer = Imputer(strategy=strategy).fit([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ])
                saved = copy.deepcopy(imputer)

                imputer.partial_fit(np.full((5, 2), np.nan))

                self.assert_learned_equal(imputer, saved)

    def test_constant_strategy_updates_valid_counts(self):
        imputer = Imputer(
            strategy="constant", fill_value=-7
        ).partial_fit([
            [np.nan, 2.0],
            [3.0, np.nan],
        ])
        imputer.partial_fit([[4.0, np.nan]])

        assert_array_equal(imputer.n_samples_seen_, [2, 1])
        assert_allclose(imputer.statistics_, [-7.0, -7.0])
        assert_allclose(
            imputer.transform([[np.nan, 8.0]]),
            [[-7.0, 8.0]],
        )

    def test_transform_does_not_learn_imputed_values(self):
        imputer = Imputer().fit([[2.0], [np.nan]])
        saved = copy.deepcopy(imputer)

        imputer.transform(np.full((100, 1), np.nan))
        self.assert_learned_equal(imputer, saved)

        imputer.partial_fit([[8.0]])

        assert_array_equal(imputer.n_samples_seen_, [2])
        assert_allclose(imputer.statistics_, [5.0])

    def test_fit_resets_and_allows_new_schema(self):
        imputer = Imputer().fit([[1.0, 2.0], [3.0, 4.0]])

        self.assertIs(imputer.fit([[20.0], [np.nan]]), imputer)
        self.assertEqual(imputer.n_features_in_, 1)
        assert_array_equal(imputer.n_samples_seen_, [1])
        assert_allclose(imputer.statistics_, [20.0])

    def test_invalid_parameters(self):
        cases = [
            ({"strategy": 1}, TypeError),
            ({"strategy": "median"}, ValueError),
            ({"fill_value": True}, TypeError),
            ({"fill_value": "0"}, TypeError),
            ({"fill_value": [0]}, TypeError),
            ({"fill_value": 1j}, TypeError),
            ({"fill_value": np.nan}, ValueError),
            ({"fill_value": np.inf}, ValueError),
            ({"fill_value": -np.inf}, ValueError),
        ]

        for parameters, error in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(error):
                    Imputer(**parameters)

    def test_invalid_learning_preserves_state(self):
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

        for strategy in ("mean", "constant"):
            imputer = Imputer(strategy=strategy).fit([
                [1.0, 2.0],
                [3.0, 4.0],
            ])
            saved = copy.deepcopy(imputer)

            for method in (imputer.fit, imputer.partial_fit):
                for X in bad_inputs:
                    with self.subTest(
                        strategy=strategy,
                        method=method.__name__,
                        X=repr(X),
                    ):
                        with self.assertRaises((TypeError, ValueError)):
                            method(X)
                        self.assert_learned_equal(imputer, saved)

            with self.assertRaises(ValueError):
                imputer.partial_fit([[1.0]])

            self.assert_learned_equal(imputer, saved)

    def test_mean_overflow_preserves_state(self):
        imputer = Imputer().fit([[1.0], [3.0]])
        saved = copy.deepcopy(imputer)

        for method in (imputer.fit, imputer.partial_fit):
            with self.subTest(method=method.__name__):
                with self.assertRaises(FloatingPointError):
                    method([[-1e308], [1e308]])
                self.assert_learned_equal(imputer, saved)

    def test_parameter_changes_require_refit(self):
        for name, value in (
            ("strategy", "constant"),
            ("fill_value", -5.0),
        ):
            with self.subTest(parameter=name):
                imputer = Imputer().fit([[1.0], [3.0]])
                saved = copy.deepcopy(imputer)

                setattr(imputer, name, value)

                with self.assertRaises(ValueError):
                    imputer.partial_fit([[5.0]])
                with self.assertRaises(ValueError):
                    imputer.transform([[np.nan]])

                self.assert_learned_equal(imputer, saved)

                imputer.fit([[np.nan]])
                expected = -5.0 if name == "fill_value" else 0.0
                assert_allclose(imputer.transform([[np.nan]]), [[expected]])

    def test_transform_validation_and_empty_input(self):
        with self.assertRaises(RuntimeError):
            Imputer().transform([[np.nan]])

        imputer = Imputer().fit([[1.0, 2.0]])
        self.assertEqual(
            imputer.transform(np.empty((0, 2))).shape,
            (0, 2),
        )

        for X in (
            [[1.0]],
            [1.0, 2.0],
            [[np.inf, 1.0]],
            [["1", "2"]],
        ):
            with self.subTest(X=X):
                with self.assertRaises((TypeError, ValueError)):
                    imputer.transform(X)

    def test_input_and_output_ownership(self):
        X = np.array([[1.0, np.nan], [3.0, 8.0]])
        original = X.copy()
        imputer = Imputer().fit(X)
        saved = copy.deepcopy(imputer)

        result = imputer.transform(X)

        assert_array_equal(X, original)
        self.assertFalse(np.shares_memory(result, X))

        X[:] = -99
        result[:] = 99

        self.assert_learned_equal(imputer, saved)

    def test_snapshot_remains_independent(self):
        imputer = Imputer().fit([[1.0], [3.0]])
        frozen = copy.deepcopy(imputer)
        expected = frozen.transform([[np.nan]])

        imputer.partial_fit([[100.0]])

        assert_array_equal(frozen.transform([[np.nan]]), expected)
        assert_array_equal(frozen.n_samples_seen_, [2])
        self.assertFalse(
            np.shares_memory(frozen.statistics_, imputer.statistics_)
        )
        self.assertFalse(
            np.shares_memory(frozen._moments[1], imputer._moments[1])
        )

    def test_read_only_noncontiguous_input(self):
        X = np.arange(48, dtype=float).reshape(8, 6)[::2, ::2]
        X[0, 1] = np.nan
        X.flags.writeable = False

        imputer = Imputer().fit(X)
        expected = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

        assert_allclose(imputer.transform(X), expected)

    def test_fit_transform_accepts_y_and_preserves_shape(self):
        imputer = Imputer()

        result = imputer.fit_transform([[1.0, np.nan], [3.0, 4.0]], [0, 1])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.dtype("float64"))
        assert_allclose(result, [[1.0, 4.0], [3.0, 4.0]])

    def test_constant_strategy_does_not_compute_moments(self):
        imputer = Imputer(strategy="constant", fill_value=-1
                    ).fit([[-1e308], [1e308], [np.nan]])

        self.assertIsNone(imputer._moments)
        assert_array_equal(imputer.n_samples_seen_, [2])
        assert_allclose(
            imputer.transform([[np.nan], [1e308]]),
            [[-1.0], [1e308]],
        )


if __name__ == "__main__":
    unittest.main()