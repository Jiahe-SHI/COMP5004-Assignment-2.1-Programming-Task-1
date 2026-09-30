"""Tests for preprocessing behavior and the shared transformer protocol."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numcompute.preprocessing import SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder
from numcompute.io import load_csv


class TestSimpleImputer(unittest.TestCase):
    def test_default_fill(self):
        assert_array_equal(SimpleImputer().fit_transform([[1, np.nan], [np.nan, 4]]), [[1, 0], [0, 4]])

    def test_custom_constant_and_all_nan_column(self):
        assert_array_equal(SimpleImputer(fill_value=-2).fit_transform([[np.nan, 1], [np.nan, np.nan]]), [[-2, 1], [-2, -2]])

    def test_fit_returns_self(self):
        imputer = SimpleImputer(fill_value=np.float32(3))
        self.assertIs(imputer.fit([[1, 2]], y=[0]), imputer)
        self.assertEqual(imputer.n_features_in_, 2)
        self.assertEqual(imputer.fill_value_, 3)

    def test_unfitted(self):
        with self.assertRaises(RuntimeError):
            SimpleImputer().transform([[1]])

    def test_wrong_feature_count(self):
        imputer = SimpleImputer().fit([[1, 2]])
        with self.assertRaises(ValueError):
            imputer.transform([[1]])

    def test_empty_shapes(self):
        for shape in [(0, 2), (2, 0), (0, 0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                SimpleImputer().fit(np.empty(shape))
        imputer = SimpleImputer().fit([[1, 2]])
        self.assertEqual(imputer.transform(np.empty((0, 2))).shape, (0, 2))
        with self.assertRaises(ValueError):
            imputer.transform(np.empty((0, 1)))

    def test_shapes_and_types(self):
        for value in [1, [1, 2], [[[1]]], [[1, 2], [3]]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimpleImputer().fit(value)
        for value in [[['1']], [[True]], [[1j]], np.array([[1]], dtype=object)]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                SimpleImputer().fit(value)

    def test_invalid_parameters(self):
        for strategy, error in [('mean', ValueError), ('median', ValueError), (None, TypeError)]:
            with self.subTest(strategy=strategy), self.assertRaises(error):
                SimpleImputer(strategy=strategy)
        for value in [True, '0', 1j, None, [0], np.array(0.)]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                SimpleImputer(fill_value=value)
        for value in [np.nan, np.inf, -np.inf, 10**400]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimpleImputer(fill_value=value)

    def test_input_unchanged_and_independent(self):
        original = np.array([[1., np.nan], [3., 4.]])
        before = original.copy()
        imputer = SimpleImputer().fit(original)
        result = imputer.transform(original)
        assert_array_equal(original, before)
        self.assertFalse(np.shares_memory(result, original))
        result[0, 0] = 99
        assert_array_equal(original, before)

    def test_noncontiguous(self):
        original = np.array([[1., 9., np.nan, 9.], [3., 9., 4., 9.]])
        X = original[:, ::2]
        self.assertFalse(X.flags.c_contiguous)
        assert_array_equal(SimpleImputer().fit_transform(X), [[1, 0], [3, 4]])
        self.assertTrue(np.isnan(original[0, 2]))

    def test_integer_output_and_non_nan_preservation(self):
        result = SimpleImputer().fit_transform([[1, 2]])
        self.assertEqual(result.dtype, np.dtype('float64'))
        assert_array_equal(result, [[1, 2]])
        assert_array_equal(SimpleImputer().fit_transform([[np.inf, -np.inf, np.nan]]), [[np.inf, -np.inf, 0]])

    def test_refit_replaces_state(self):
        imputer = SimpleImputer(fill_value=-1).fit([[1, 2]])
        imputer.fill_value = 5
        assert_array_equal(imputer.transform([[np.nan, 2]]), [[-1, 2]])
        imputer.fit([[1, 2, 3]])
        self.assertEqual(imputer.n_features_in_, 3)
        assert_array_equal(imputer.transform([[np.nan, 2, 3]]), [[5, 2, 3]])

    def test_failed_refit_preserves_state(self):
        imputer = SimpleImputer(fill_value=-1).fit([[1, 2]])
        with self.assertRaises(ValueError):
            imputer.fit(np.empty((0, 3)))
        assert_array_equal(imputer.transform([[np.nan, 2]]), [[-1, 2]])
        imputer.fill_value = np.nan
        with self.assertRaises(ValueError):
            imputer.fit([[1, 2]])
        assert_array_equal(imputer.transform([[np.nan, 2]]), [[-1, 2]])

    def test_fit_transform_equivalence(self):
        X = [[1, np.nan], [2, 3]]
        assert_array_equal(SimpleImputer(fill_value=7).fit_transform(X),
                           SimpleImputer(fill_value=7).fit(X).transform(X))

    def test_csv_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'data.csv'
            path.write_text('age,hours,city\n20,,A\nNA,6,B\n', encoding='utf-8')
            X = load_csv(path, skip_header=1, usecols=[0, 1])
            assert_array_equal(SimpleImputer(fill_value=-1).fit_transform(X), [[20, -1], [-1, 6]])

class TestStandardScaler(unittest.TestCase):

    def test_fit_statistics(self):
        X = [[1, 10], [3, 14]]
        scaler = StandardScaler()

        self.assertIs(scaler.fit(X), scaler)
        assert_allclose(scaler.mean_, [2, 12])
        assert_allclose(scaler.scale_, [1, 2])
        self.assertEqual(scaler.n_features_in_, 2)

    def test_standardized_values(self):
        X = [[1, 10], [3, 14]]

        result = StandardScaler().fit_transform(X)

        assert_allclose(result, [[-1, -1], [1, 1]])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_constant_column(self):
        X = [[5, 1], [5, 3]]
        scaler = StandardScaler().fit(X)

        assert_allclose(scaler.scale_, [1, 1])
        assert_allclose(
            scaler.transform(X),
            [[0, -1], [0, 1]],
        )

    def test_nan_handling(self):
        X = [
            [1, np.nan],
            [3, 10],
            [np.nan, 14],
        ]
        scaler = StandardScaler().fit(X)

        assert_allclose(scaler.mean_, [2, 12])
        assert_allclose(scaler.scale_, [1, 2])

        assert_allclose(
            scaler.transform(X),
            [
                [-1, np.nan],
                [1, -1],
                [np.nan, 1],
            ],
            equal_nan=True,
        )

    def test_all_nan_training_column(self):
        with self.assertRaises(ValueError):
            StandardScaler().fit([
                [1, np.nan],
                [2, np.nan],
            ])

    def test_infinity_rejected(self):
        scaler = StandardScaler().fit([[1], [3]])

        for value in [np.inf, -np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    StandardScaler().fit([[value]])

                with self.assertRaises(ValueError):
                    scaler.transform([[value]])

    def test_unfitted(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform([[1, 2]])

    def test_wrong_feature_count(self):
        scaler = StandardScaler().fit([[1, 2]])

        with self.assertRaises(ValueError):
            scaler.transform([[1, 2, 3]])

    def test_empty_inputs(self):
        for shape in [(0, 2), (2, 0), (0, 0)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    StandardScaler().fit(np.empty(shape))

        scaler = StandardScaler().fit([[1, 2]])

        result = scaler.transform(np.empty((0, 2)))
        self.assertEqual(result.shape, (0, 2))

        with self.assertRaises(ValueError):
            scaler.transform(np.empty((0, 3)))

    def test_invalid_shapes_and_types(self):
        scaler = StandardScaler().fit([[1, 2]])

        for X in [1, [1, 2], [[[1, 2]]], [[1, 2], [3]]]:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    scaler.fit(X)

                with self.assertRaises(ValueError):
                    scaler.transform(X)

        for X in [
            [["1", "2"]],
            [[True, False]],
            [[1j, 2j]],
            np.array([[1, 2]], dtype=object),
        ]:
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    scaler.fit(X)

                with self.assertRaises(TypeError):
                    scaler.transform(X)

    def test_input_unchanged(self):
        X = np.array([[1., np.nan], [3., 4.]])
        before = X.copy()

        result = StandardScaler().fit_transform(X)

        assert_array_equal(X, before)
        self.assertFalse(np.shares_memory(X, result))

        result[0, 0] = 999
        assert_array_equal(X, before)

    def test_noncontiguous_input(self):
        original = np.array([
            [1., 99., 10., 99.],
            [3., 99., 14., 99.],
        ])
        X = original[:, ::2]

        self.assertFalse(X.flags.c_contiguous)

        result = StandardScaler().fit_transform(X)

        assert_allclose(result, [[-1, -1], [1, 1]])

    def test_transform_uses_training_statistics(self):
        scaler = StandardScaler().fit([[1], [3]])

        result = scaler.transform([[5], [7]])

        # Training mean=2, std=1; do not refit on the new data.
        assert_allclose(result, [[3], [5]])
        assert_allclose(scaler.mean_, [2])
        assert_allclose(scaler.scale_, [1])

    def test_refit_replaces_state(self):
        scaler = StandardScaler().fit([[1], [3]])

        scaler.fit([[10, 100], [14, 108]])

        self.assertEqual(scaler.n_features_in_, 2)
        assert_allclose(scaler.mean_, [12, 104])
        assert_allclose(scaler.scale_, [2, 4])
        assert_allclose(scaler.transform([[12, 104]]), [[0, 0]])

    def test_failed_refit_preserves_state(self):
        scaler = StandardScaler().fit([[1], [3]])

        with self.assertRaises(ValueError):
            scaler.fit([[np.nan, 1], [np.nan, 2]])

        self.assertEqual(scaler.n_features_in_, 1)
        assert_allclose(scaler.mean_, [2])
        assert_allclose(scaler.scale_, [1])
        assert_allclose(scaler.transform([[5]]), [[3]])

    def test_fit_transform_equivalence(self):
        X = [[1, 10], [3, 14]]

        combined = StandardScaler().fit_transform(X, y=[0, 1])
        separate = StandardScaler().fit(X).transform(X)

        assert_allclose(combined, separate)

    def test_single_sample(self):
        scaler = StandardScaler().fit([[5, 10]])

        assert_allclose(scaler.scale_, [1, 1])
        assert_allclose(scaler.transform([[5, 10]]), [[0, 0]])

    def test_all_nan_transform_allowed(self):
        scaler = StandardScaler().fit([[1, 10], [3, 14]])

        result = scaler.transform([[np.nan, np.nan]])

        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.all(np.isnan(result)))

    def test_numeric_overflow_rejected(self):
        with self.assertRaises(ValueError):
            StandardScaler().fit([[-1e308], [1e308]])

        scaler = StandardScaler().fit([[-1e308]])

        with self.assertRaises(ValueError):
            scaler.transform([[1e308]])

    def test_imputer_scaler_integration(self):
        X = [[1, np.nan], [3, 4]]

        filled = SimpleImputer(fill_value=0).fit_transform(X)
        result = StandardScaler().fit_transform(filled)

        assert_allclose(result, [[-1, -1], [1, 1]])

class TestMinMaxScaler(unittest.TestCase):

    def test_fit_statistics(self):
        scaler = MinMaxScaler()

        self.assertIs(scaler.fit([[10, 2], [30, 6]]), scaler)
        assert_allclose(scaler.data_min_, [10, 2])
        assert_allclose(scaler.data_max_, [30, 6])
        self.assertEqual(scaler.n_features_in_, 2)

    def test_default_range(self):
        X = [[10, 2], [20, 4], [30, 6]]

        result = MinMaxScaler().fit_transform(X)

        assert_allclose(result, [[0, 0], [0.5, 0.5], [1, 1]])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_custom_range(self):
        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform([[10], [20], [30]])

        assert_allclose(result, [[-1], [0], [1]])

    def test_constant_column(self):
        scaler = MinMaxScaler(feature_range=(2, 4))
        X = [[5, 10], [5, 20]]

        result = scaler.fit_transform(X)

        assert_allclose(result, [[2, 2], [2, 4]])

        # Constant column uses an effective denominator of 1.
        assert_allclose(scaler.transform([[6, 15]]), [[4, 3]])

    def test_nan_handling(self):
        X = [
            [10, np.nan],
            [30, 2],
            [np.nan, 6],
        ]
        scaler = MinMaxScaler().fit(X)

        assert_allclose(scaler.data_min_, [10, 2])
        assert_allclose(scaler.data_max_, [30, 6])
        assert_allclose(scaler.transform(X),[[0, np.nan], [1, 0], [np.nan, 1]],equal_nan=True,)

    def test_all_nan_training_column(self):
        with self.assertRaises(ValueError):
            MinMaxScaler().fit([
                [1, np.nan],
                [2, np.nan],
            ])

    def test_infinity_rejected(self):
        scaler = MinMaxScaler().fit([[1], [3]])

        for value in [np.inf, -np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MinMaxScaler().fit([[value]])

                with self.assertRaises(ValueError):
                    scaler.transform([[value]])

    def test_invalid_range_types(self):
        for bounds in [None, "0,1", 1]:
            with self.subTest(bounds=bounds):
                with self.assertRaises(TypeError):
                    MinMaxScaler(feature_range=bounds)

        for bounds in [(False, 1), ("0", 1), (0, 1j)]:
            with self.subTest(bounds=bounds):
                with self.assertRaises(TypeError):
                    MinMaxScaler(feature_range=bounds)

    def test_invalid_range_values(self):
        for bounds in [
            (),
            (0,),
            (0, 1, 2),
            (1, 1),
            (2, 1),
            (np.nan, 1),
            (0, np.inf),
            (-np.inf, 1),
            (0, 10**400),
            (-1e308, 1e308),
        ]:
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    MinMaxScaler(feature_range=bounds)

    def test_unfitted(self):
        with self.assertRaises(RuntimeError):
            MinMaxScaler().transform([[1, 2]])

    def test_wrong_feature_count(self):
        scaler = MinMaxScaler().fit([[1, 2]])

        with self.assertRaises(ValueError):
            scaler.transform([[1, 2, 3]])

    def test_empty_inputs(self):
        for shape in [(0, 2), (2, 0), (0, 0)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    MinMaxScaler().fit(np.empty(shape))

        scaler = MinMaxScaler().fit([[1, 2]])

        self.assertEqual(
            scaler.transform(np.empty((0, 2))).shape,
            (0, 2),
        )

        with self.assertRaises(ValueError):
            scaler.transform(np.empty((0, 3)))

    def test_invalid_shapes_and_types(self):
        scaler = MinMaxScaler().fit([[1, 2]])

        for X in [1, [1, 2], [[[1, 2]]], [[1, 2], [3]]]:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    scaler.fit(X)

                with self.assertRaises(ValueError):
                    scaler.transform(X)

        for X in [
            [["1", "2"]],
            [[True, False]],
            [[1j, 2j]],
            np.array([[1, 2]], dtype=object),
        ]:
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    scaler.fit(X)

                with self.assertRaises(TypeError):
                    scaler.transform(X)

    def test_input_unchanged(self):
        X = np.array([[10., np.nan], [30., 4.]])
        before = X.copy()

        result = MinMaxScaler().fit_transform(X)

        assert_array_equal(X, before)
        self.assertFalse(np.shares_memory(X, result))

        result[0, 0] = 999
        assert_array_equal(X, before)

    def test_noncontiguous_input(self):
        original = np.array([
            [10., 99., 2., 99.],
            [30., 99., 6., 99.],
        ])
        X = original[:, ::2]

        self.assertFalse(X.flags.c_contiguous)

        assert_allclose(MinMaxScaler().fit_transform(X),[[0, 0], [1, 1]],)

    def test_new_data_not_clipped_or_refitted(self):
        scaler = MinMaxScaler().fit([[10], [30]])

        result = scaler.transform([[0], [40]])

        assert_allclose(result, [[-0.5], [1.5]])
        assert_allclose(scaler.data_min_, [10])
        assert_allclose(scaler.data_max_, [30])

    def test_refit_replaces_state(self):
        scaler = MinMaxScaler().fit([[10], [30]])
        scaler.feature_range = (-1, 1)

        # Parameter changes take effect only on the next fit.
        assert_allclose(scaler.transform([[20]]), [[0.5]])

        scaler.fit([[0, 10], [4, 30]])

        self.assertEqual(scaler.n_features_in_, 2)
        assert_allclose(scaler.data_min_, [0, 10])
        assert_allclose(scaler.data_max_, [4, 30])
        assert_allclose(scaler.transform([[2, 20]]), [[0, 0]])

    def test_failed_refit_preserves_state(self):
        scaler = MinMaxScaler().fit([[10], [30]])

        with self.assertRaises(ValueError):
            scaler.fit([[np.nan, 1], [np.nan, 2]])

        self.assertEqual(scaler.n_features_in_, 1)
        assert_allclose(scaler.transform([[20]]), [[0.5]])

        scaler.feature_range = (1, 1)

        with self.assertRaises(ValueError):
            scaler.fit([[0], [4]])

        assert_allclose(scaler.transform([[20]]), [[0.5]])

    def test_fit_transform_equivalence(self):
        X = [[10, 2], [30, 6]]

        combined = MinMaxScaler().fit_transform(X, y=[0, 1])
        separate = MinMaxScaler().fit(X).transform(X)

        assert_allclose(combined, separate)

    def test_single_sample(self):
        scaler = MinMaxScaler(feature_range=(-2, 2))

        assert_allclose(
            scaler.fit_transform([[5, 10]]),
            [[-2, -2]],
        )

    def test_all_nan_transform_allowed(self):
        scaler = MinMaxScaler().fit([[10, 2], [30, 6]])

        result = scaler.transform([[np.nan, np.nan]])

        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.all(np.isnan(result)))

    def test_numeric_overflow_rejected(self):
        with self.assertRaises(ValueError):
            MinMaxScaler().fit([[-1e308], [1e308]])

        scaler = MinMaxScaler().fit([[-1e308]])

        with self.assertRaises(ValueError):
            scaler.transform([[1e308]])

    def test_imputer_scaler_integration(self):
        X = [[10, np.nan], [30, 4]]

        filled = SimpleImputer(fill_value=0).fit_transform(X)
        result = MinMaxScaler().fit_transform(filled)

        assert_allclose(result, [[0, 0], [1, 1]])

class TestOneHotEncoder(unittest.TestCase):

    def test_first_appearance_order(self):
        X = [["Sydney"], ["Adelaide"], ["Sydney"], ["Melbourne"]]
        encoder = OneHotEncoder()

        self.assertIs(encoder.fit(X), encoder)
        self.assertEqual(encoder.n_features_in_, 1)
        assert_array_equal(
            encoder.categories_[0],
            ["Sydney", "Adelaide", "Melbourne"],
        )
        assert_array_equal(
            encoder.transform(X),
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
        )

    def test_multiple_columns(self):
        X = [
            ["red", "small"],
            ["blue", "large"],
            ["red", "large"],
        ]

        result = OneHotEncoder().fit_transform(X)

        # First block: red, blue. Second block: small, large.
        assert_array_equal(
            result,
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
            ],
        )
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_numeric_categories(self):
        X = [[2], [1], [2.0], [3]]
        encoder = OneHotEncoder().fit(X)

        assert_array_equal(encoder.categories_[0], [2, 1, 3])
        assert_array_equal(
            encoder.transform(X),
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
        )

    def test_different_types_in_separate_columns(self):
        X = [["red", 1], ["blue", 2]]

        result = OneHotEncoder().fit_transform(X)

        assert_array_equal(result, [[1, 0, 1, 0], [0, 1, 0, 1]])

    def test_none_and_nan_share_category(self):
        X = [["red"], [None], [np.nan], ["blue"]]
        encoder = OneHotEncoder().fit(X)

        assert_array_equal(
            encoder.categories_[0],
            np.array(["red", None, "blue"], dtype=object),
        )
        assert_array_equal(
            encoder.transform(X),
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        )

    def test_all_missing_training_column(self):
        encoder = OneHotEncoder().fit([[None], [np.nan]])

        assert_array_equal(
            encoder.categories_[0],
            np.array([None], dtype=object),
        )
        assert_array_equal(
            encoder.transform([[np.nan], [None]]),
            [[1], [1]],
        )

    def test_string_nan_is_a_regular_category(self):
        encoder = OneHotEncoder().fit([["nan"], [None]])

        self.assertEqual(len(encoder.categories_[0]), 2)
        assert_array_equal(
            encoder.transform([["nan"], [np.nan]]),
            [[1, 0], [0, 1]],
        )

    def test_unknown_error(self):
        encoder = OneHotEncoder().fit([["red"], ["blue"]])

        with self.assertRaises(ValueError):
            encoder.transform([["green"]])

    def test_unknown_ignore_only_affected_block(self):
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit([["red", "small"], ["blue", "large"]])

        result = encoder.transform([
            ["green", "small"],
            ["red", "medium"],
        ])

        assert_array_equal(
            result,
            [[0, 0, 1, 0], [1, 0, 0, 0]],
        )

    def test_unseen_missing_category(self):
        for missing in [None, np.nan]:
            with self.subTest(missing=missing):
                encoder = OneHotEncoder().fit([["red"]])

                with self.assertRaises(ValueError):
                    encoder.transform([[missing]])

                encoder = OneHotEncoder(
                    handle_unknown="ignore"
                ).fit([["red"]])

                assert_array_equal(
                    encoder.transform([[missing]]),
                    [[0]],
                )

    def test_mixed_types_in_one_column_rejected(self):
        encoder = OneHotEncoder().fit([["red"]])

        for X in [[["red"], [1]], [[1], ["red"]]]:
            with self.subTest(X=X):
                with self.assertRaises(TypeError):
                    OneHotEncoder().fit(X)

                with self.assertRaises(TypeError):
                    encoder.transform(X)

    def test_invalid_category_types(self):
        encoder = OneHotEncoder().fit([[1]])

        for value in [True, np.bool_(False), 1j, {"a": 1}]:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    OneHotEncoder().fit([[value]])

                with self.assertRaises(TypeError):
                    encoder.transform([[value]])

    def test_infinity_rejected(self):
        encoder = OneHotEncoder().fit([[1]])

        for value in [np.inf, -np.inf]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    OneHotEncoder().fit([[value]])

                with self.assertRaises(ValueError):
                    encoder.transform([[value]])

    def test_invalid_option(self):
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="skip")

        for option in [None, True, 1]:
            with self.subTest(option=option):
                with self.assertRaises(TypeError):
                    OneHotEncoder(handle_unknown=option)

    def test_unfitted(self):
        with self.assertRaises(RuntimeError):
            OneHotEncoder().transform([["red"]])

    def test_wrong_feature_count(self):
        encoder = OneHotEncoder().fit([["red", "small"]])

        with self.assertRaises(ValueError):
            encoder.transform([["red"]])

    def test_empty_inputs(self):
        for shape in [(0, 2), (2, 0), (0, 0)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    OneHotEncoder().fit(
                        np.empty(shape, dtype=object)
                    )

        encoder = OneHotEncoder().fit([
            ["red", "small"],
            ["blue", "large"],
        ])

        result = encoder.transform(
            np.empty((0, 2), dtype=object)
        )

        self.assertEqual(result.shape, (0, 4))
        self.assertEqual(result.dtype, np.dtype("float64"))

        with self.assertRaises(ValueError):
            encoder.transform(np.empty((0, 1), dtype=object))

    def test_invalid_shapes(self):
        encoder = OneHotEncoder().fit([["red"]])

        for X in [
            "red",
            ["red", "blue"],
            [[["red"]]],
            [["red", "small"], ["blue"]],
        ]:
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    OneHotEncoder().fit(X)

                with self.assertRaises(ValueError):
                    encoder.transform(X)

    def test_input_unchanged(self):
        X = np.array([["red"], [None]], dtype=object)
        before = X.copy()

        result = OneHotEncoder().fit_transform(X)

        assert_array_equal(X, before)
        self.assertFalse(np.shares_memory(X, result))

        result[0, 0] = 99
        assert_array_equal(X, before)

    def test_noncontiguous_input(self):
        original = np.array([
            ["red", "unused", "small", "unused"],
            ["blue", "unused", "large", "unused"],
        ], dtype=object)
        X = original[:, ::2]

        self.assertFalse(X.flags.c_contiguous)

        assert_array_equal(
            OneHotEncoder().fit_transform(X),
            [[1, 0, 1, 0], [0, 1, 0, 1]],
        )

    def test_refit_replaces_state(self):
        encoder = OneHotEncoder().fit([["red"], ["blue"]])

        encoder.fit([["small", 1], ["large", 1]])

        self.assertEqual(encoder.n_features_in_, 2)
        assert_array_equal(
            encoder.categories_[0],
            ["small", "large"],
        )
        assert_array_equal(encoder.categories_[1], [1])
        assert_array_equal(
            encoder.transform([["large", 1]]),
            [[0, 1, 1]],
        )

        with self.assertRaises(ValueError):
            encoder.transform([["red", 1]])

    def test_failed_refit_preserves_state(self):
        encoder = OneHotEncoder().fit([["red"], ["blue"]])

        with self.assertRaises(TypeError):
            encoder.fit([["red"], [1]])

        assert_array_equal(
            encoder.categories_[0],
            ["red", "blue"],
        )
        assert_array_equal(
            encoder.transform([["blue"]]),
            [[0, 1]],
        )

    def test_option_change_requires_refit(self):
        encoder = OneHotEncoder().fit([["red"]])
        encoder.handle_unknown = "ignore"

        with self.assertRaises(ValueError):
            encoder.transform([["blue"]])

        encoder.fit([["red"]])

        assert_array_equal(
            encoder.transform([["blue"]]),
            [[0]],
        )

    def test_transform_does_not_learn_unknowns(self):
        encoder = OneHotEncoder(
            handle_unknown="ignore"
        ).fit([["red"]])

        assert_array_equal(
            encoder.transform([["blue"]]),
            [[0]],
        )
        assert_array_equal(encoder.categories_[0], ["red"])
        assert_array_equal(
            encoder.transform([["red"]]),
            [[1]],
        )

    def test_fit_transform_equivalence(self):
        X = [["red", 1], ["blue", 2], ["red", 2]]

        combined = OneHotEncoder().fit_transform(X, y=[0, 1, 0])
        separate = OneHotEncoder().fit(X).transform(X)

        assert_array_equal(combined, separate)

    def test_csv_encoder_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "categories.csv"
            path.write_text(
                "city\nAdelaide\nMelbourne\nAdelaide\n",
                encoding="utf-8",
            )

            X = load_csv(path, skip_header=1, dtype=str)
            result = OneHotEncoder().fit_transform(X)

            assert_array_equal(result, [[1, 0], [0, 1], [1, 0]])

if __name__ == '__main__':
    unittest.main()
