import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.utils import (_as_float_array, _validate_integer, 
sigmoid, softmax, logsumexp, euclidean_distances, batch_iter)

class TestInputValidation(unittest.TestCase):

    def test_numeric_input_converts_to_float64(self):
        result = _as_float_array([[1, 2], [3, 4]], ndim=2)

        self.assertEqual(result.dtype, np.dtype("float64"))
        self.assertEqual(result.shape, (2, 2))
        assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_result_does_not_share_input_memory(self):
        original = np.array([1.0, 2.0, 3.0])
        result = _as_float_array(original)

        self.assertFalse(np.shares_memory(original, result))
        result[0] = 99.0
        assert_array_equal(original, [1.0, 2.0, 3.0])

    def test_non_contiguous_input(self):
        original = np.arange(12).reshape(3, 4)
        sliced = original[:, ::2]

        self.assertFalse(sliced.flags.c_contiguous)
        result = _as_float_array(sliced, ndim=2)

        assert_array_equal(result, [[0, 2], [4, 6], [8, 10]])

    def test_unsupported_numeric_types(self):
        invalid_inputs = [
            [True, False],
            [True, 2],
            ["1", "2"],
            [1, "2"],
            [1 + 2j],
            np.array([1, 2], dtype=object),
        ]

        for values in invalid_inputs:
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    _as_float_array(values)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            _as_float_array([1, 2, 3], ndim=2)

        with self.assertRaises(ValueError):
            _as_float_array([[1, 2], [3]])

    def test_special_values_follow_finite_option(self):
        values = np.array([np.nan, np.inf, -np.inf])
        result = _as_float_array(values)

        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], np.inf)
        self.assertEqual(result[2], -np.inf)

        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _as_float_array([value], finite=True)

    def test_scalar_and_empty_input(self):
        scalar = _as_float_array(3)
        empty = _as_float_array([])

        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.item(), 3.0)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype, np.dtype("float64"))

    def test_integer_validation(self):
        for value in (3, np.int64(3)):
            with self.subTest(value=value):
                result = _validate_integer(value, name="batch_size", minimum=1)
                self.assertEqual(result, 3)
                self.assertIs(type(result), int)

        for value in (True, np.bool_(False), 3.0, "3"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    _validate_integer(value, name="batch_size", minimum=1)

        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _validate_integer(value, name="batch_size", minimum=1)

class TestSigmoid(unittest.TestCase):

    def test_known_values(self):
        values = [-np.log(3.0), 0.0, np.log(3.0)]

        result = sigmoid(values)

        assert_allclose(
            result,
            [0.25, 0.5, 0.75],
            rtol=1e-14,
            atol=1e-14,
        )
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_extreme_finite_values(self):
        values = [-1e308, -1000.0, 1000.0, 1e308]

        # Overflow, invalid operations and division by zero must fail.
        with np.errstate(
            over="raise", invalid="raise", divide="raise"
        ):
            result = sigmoid(values)

        assert_array_equal(result, [0.0, 0.0, 1.0, 1.0])

    def test_nan_and_infinity(self):
        result = sigmoid([-np.inf, 0.0, np.inf, np.nan])

        assert_array_equal(result[:3], [0.0, 0.5, 1.0])
        self.assertTrue(np.isnan(result[3]))

    def test_scalar_and_empty_input(self):
        scalar = sigmoid(0)
        empty = sigmoid(np.empty((0, 2)))

        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.item(), 0.5)
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty.dtype, np.dtype("float64"))

    def test_non_contiguous_input_is_preserved(self):
        original = np.array([
            [-2.0, 99.0, 0.0, 99.0],
            [1.0, 99.0, 2.0, 99.0],
        ])
        before = original.copy()
        sliced = original[:, ::2]

        self.assertFalse(sliced.flags.c_contiguous)

        result = sigmoid(sliced)

        # The direct formula is safe for these small reference values.
        expected = 1.0 / (1.0 + np.exp(-sliced))
        assert_allclose(result, expected)
        self.assertEqual(result.shape, sliced.shape)
        self.assertFalse(np.shares_memory(result, original))
        assert_array_equal(original, before)

    def test_invalid_types_are_rejected(self):
        for values in ([True, 1], ["1", "2"], [1 + 2j]):
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    sigmoid(values)

class TestSoftmax(unittest.TestCase):

    def test_known_probabilities(self):
        values = np.log([1.0, 2.0, 3.0])

        result = softmax(values)

        assert_allclose(result, [1 / 6, 2 / 6, 3 / 6])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_axis_and_negative_axis(self):
        values = np.array([[0.0, 0.0],[0.0, np.log(3.0)],])

        rows = softmax(values, axis=1)
        columns = softmax(values, axis=0)

        assert_allclose(rows, [[0.5, 0.5], [0.25, 0.75]])
        assert_allclose(columns, [[0.5, 0.25], [0.5, 0.75]])
        assert_allclose(softmax(values, axis=-1), rows)
        assert_allclose(softmax(values, axis=np.int64(-2)), columns)

    def test_extreme_finite_values(self):
        values = [
            [-1e308, 1e308],
            [1e308, 1e308],
            [-1e308, -1e308],
        ]

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = softmax(values)

        assert_allclose(result, [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]],)

    def test_special_values_per_slice(self):
        values = [
            [np.inf, 0.0, np.inf],
            [-np.inf, -np.inf, -np.inf],
            [np.nan, np.inf, 0.0],
            [-np.inf, 0.0, 0.0],
        ]

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = softmax(values, axis=1)

        expected = [
            [0.5, 0.0, 0.5],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0.0, 0.5, 0.5],
        ]
        assert_allclose(result, expected, equal_nan=True)

    def test_non_contiguous_input_is_preserved(self):
        original = np.arange(24.0).reshape(2, 3, 4)
        before = original.copy()
        sliced = original[:, :, ::2]

        self.assertFalse(sliced.flags.c_contiguous)

        result = softmax(sliced, axis=-1)

        # The direct formula is safe for these small values.
        exponentials = np.exp(sliced)
        expected = exponentials / exponentials.sum(axis=-1, keepdims=True)

        assert_allclose(result, expected)
        assert_allclose(result.sum(axis=-1), np.ones((2, 3)))
        self.assertEqual(result.shape, sliced.shape)
        self.assertFalse(np.shares_memory(result, original))
        assert_array_equal(original, before)

    def test_empty_axis_and_scalar(self):
        with self.assertRaises(ValueError):
            softmax([])

        with self.assertRaises(ValueError):
            softmax(np.empty((2, 0)), axis=1)

        with self.assertRaises(ValueError):
            softmax(3.0)

        # Zero rows are allowed when the reduction axis is non-empty.
        result = softmax(np.empty((0, 3)), axis=1)
        self.assertEqual(result.shape, (0, 3))

    def test_invalid_axis(self):
        values = [[1.0, 2.0], [3.0, 4.0]]

        for axis in (True, 1.0, "1", None, (0, 1)):
            with self.subTest(axis=axis):
                with self.assertRaises(TypeError):
                    softmax(values, axis=axis)

        for axis in (2, -3):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    softmax(values, axis=axis)

    def test_invalid_input_types(self):
        for values in ([True, 1], ["1", "2"], [1 + 2j]):
            with self.subTest(values=values):
                with self.assertRaises(TypeError):
                    softmax(values)

class TestLogSumExp(unittest.TestCase):

    def test_known_values(self):
        result = logsumexp([0.0, 0.0])

        assert_allclose(result, np.log(2.0))
        self.assertIsInstance(result, np.float64)

        assert_allclose(logsumexp(np.log([1.0, 2.0, 3.0])),np.log(6.0),)

    def test_axis_and_keepdims(self):
        values = np.log([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        assert_allclose(logsumexp(values), np.log(10.0))
        assert_allclose(logsumexp(values, axis=0),np.log([4.0, 6.0]),)
        assert_allclose(logsumexp(values, axis=-1),np.log([3.0, 7.0]),)

        rows = logsumexp(values, axis=1, keepdims=True)
        self.assertEqual(rows.shape, (2, 1))
        assert_allclose(rows, np.log([[3.0], [7.0]]))

        total = logsumexp(values, keepdims=True)
        self.assertEqual(total.shape, (1, 1))
        assert_allclose(total, [[np.log(10.0)]])

    def test_extreme_finite_values(self):
        values = [
            [1000.0, 1000.0],
            [-1000.0, -1000.0],
            [-1e308, 1e308],
        ]

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = logsumexp(values, axis=1)

        expected = [1000.0 + np.log(2.0),-1000.0 + np.log(2.0),1e308,]
        assert_allclose(result, expected)

    def test_special_values_per_slice(self):
        values = [
            [np.inf, 0.0],
            [-np.inf, -np.inf],
            [np.nan, np.inf],
            [-np.inf, 2.0],
        ]

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = logsumexp(values, axis=1)

        assert_allclose(result,[np.inf, -np.inf, np.nan, 2.0],equal_nan=True,)

    def test_empty_inputs(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self.assertEqual(logsumexp([]), -np.inf)

            columns = logsumexp(np.empty((0, 3)), axis=0)
            assert_array_equal(columns, [-np.inf, -np.inf, -np.inf])

            rows = logsumexp(np.empty((0, 3)), axis=1)
            self.assertEqual(rows.shape, (0,))

            kept = logsumexp(np.empty((2, 0)),axis=1,keepdims=True,)
            self.assertEqual(kept.shape, (2, 1))
            assert_array_equal(kept, [[-np.inf], [-np.inf]])

    def test_scalar_input(self):
        self.assertEqual(logsumexp(5.0), 5.0)
        self.assertEqual(logsumexp(-3.0), -3.0)

        # A scalar has no axis to select.
        with self.assertRaises(ValueError):
            logsumexp(5.0, axis=0)

    def test_non_contiguous_input_is_preserved(self):
        original = np.arange(24.0).reshape(2, 3, 4)
        before = original.copy()
        sliced = original[:, :, ::2]

        self.assertFalse(sliced.flags.c_contiguous)

        result = logsumexp(sliced, axis=-1)

        # The direct formula is safe for these small values.
        expected = np.log(np.sum(np.exp(sliced), axis=-1))
        assert_allclose(result, expected)
        self.assertEqual(result.shape, (2, 3))
        assert_array_equal(original, before)

    def test_invalid_parameters_and_types(self):
        values = [[1.0, 2.0], [3.0, 4.0]]

        for axis in (True, 1.0, "1", (0, 1)):
            with self.subTest(axis=axis):
                with self.assertRaises(TypeError):
                    logsumexp(values, axis=axis)

        for axis in (2, -3):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    logsumexp(values, axis=axis)

        for keepdims in (1, "True", None):
            with self.subTest(keepdims=keepdims):
                with self.assertRaises(TypeError):
                    logsumexp(values, keepdims=keepdims)

        for invalid in ([True, 1], ["1", "2"], [1 + 2j]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    logsumexp(invalid)

class TestEuclideanDistances(unittest.TestCase):

    def test_known_distances(self):
        X = [[0, 0], [3, 4]]
        Y = [[0, 4], [3, 0], [0, 0]]

        result = euclidean_distances(X, Y)

        assert_allclose(result,[[4.0, 3.0, 0.0], [3.0, 4.0, 5.0]],)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_self_distances(self):
        X = [[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]

        result = euclidean_distances(X)

        assert_allclose(
            result,
            [
                [0.0, 5.0, 10.0],
                [5.0, 0.0, 5.0],
                [10.0, 5.0, 0.0],
            ],
        )
        assert_array_equal(result, result.T)

    def test_integer_conversion_before_subtraction(self):
        X = np.array([[30000, 0]], dtype=np.int16)
        Y = np.array([[-30000, 0]], dtype=np.int16)

        result = euclidean_distances(X, Y)

        assert_allclose(result, [[60000.0]])

    def test_nearby_large_coordinates(self):
        X = [[1e12, 1e12]]
        Y = [[1e12 + 3.0, 1e12 + 4.0]]

        result = euclidean_distances(X, Y)

        assert_allclose(result, [[5.0]], rtol=1e-14)

    def test_extreme_scales(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            large = euclidean_distances([[1e308, 1e308]], [[0.0, 0.0]])
            small = euclidean_distances([[1e-300, 1e-300]], [[0.0, 0.0]])
            beyond_range = euclidean_distances([[1e308]], [[-1e308]])

        # Compare after rescaling so tiny values cannot pass as zero.
        assert_allclose(large / 1e308,[[np.sqrt(2.0)]],rtol=1e-14,atol=0.0,)
        assert_allclose(small / 1e-300,[[np.sqrt(2.0)]],rtol=1e-14,atol=0.0,)
        self.assertTrue(np.isposinf(beyond_range[0, 0]))

    def test_zero_samples(self):
        empty = np.empty((0, 2))
        samples = [[1.0, 2.0], [3.0, 4.0]]

        self.assertEqual(euclidean_distances(empty, samples).shape,(0, 2),)
        self.assertEqual(euclidean_distances(samples, empty).shape,(2, 0),)
        self.assertEqual(euclidean_distances(empty).shape,(0, 0),)

    def test_invalid_inputs(self):
        invalid_shapes = [([1.0, 2.0], None),
                          ([[1.0, 2.0]], [1.0, 2.0]),
                          (np.empty((2, 0)), None),
                          ([[1.0, 2.0]], [[1.0]]),]

        for X, Y in invalid_shapes:
            with self.subTest(X=X, Y=Y):
                with self.assertRaises(ValueError):
                    euclidean_distances(X, Y)

        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    euclidean_distances([[value]])

                with self.assertRaises(ValueError):
                    euclidean_distances([[0.0]], [[value]])

        for invalid in ([[True]], [["1"]], [[1 + 2j]]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    euclidean_distances(invalid)

                with self.assertRaises(TypeError):
                    euclidean_distances([[0.0]], invalid)

    def test_blocks_non_contiguous_and_input_preservation(self):
        # Both inputs exceed the block size of 64.
        X_base = np.arange(268.0).reshape(67, 4)
        Y_base = np.arange(264.0).reshape(66, 4)
        X_before = X_base.copy()
        Y_before = Y_base.copy()

        X = X_base[:, ::2]
        Y = Y_base[:, ::2]

        self.assertFalse(X.flags.c_contiguous)
        self.assertFalse(Y.flags.c_contiguous)

        result = euclidean_distances(X, Y)

        # Direct squaring is safe for these small reference values.
        differences = X[:, None, :] - Y[None, :, :]
        expected = np.sqrt(
            np.sum(differences ** 2, axis=2)
        )

        self.assertEqual(result.shape, (67, 66))
        assert_allclose(result, expected)
        assert_array_equal(X_base, X_before)
        assert_array_equal(Y_base, Y_before)

class TestBatchIter(unittest.TestCase):

    def test_order_and_final_smaller_batch(self):
        X = np.arange(14).reshape(7, 2)

        iterator = batch_iter(X, batch_size=3)
        self.assertIs(iter(iterator), iterator)
        batches = list(iterator)

        self.assertEqual(
            [batch.shape for batch in batches],
            [(3, 2), (3, 2), (1, 2)],
        )
        assert_array_equal(batches[0], X[:3])
        assert_array_equal(batches[1], X[3:6])
        assert_array_equal(batches[2], X[6:])
        assert_array_equal(np.concatenate(batches), X)

    def test_drop_last_and_exact_batches(self):
        batches = list(batch_iter(np.arange(7), 3, drop_last=True))
        self.assertEqual(len(batches), 2)
        assert_array_equal(np.concatenate(batches), np.arange(6))

        exact = list(batch_iter(np.arange(6), 3, drop_last=True))
        self.assertEqual(len(exact), 2)
        assert_array_equal(np.concatenate(exact), np.arange(6))

        self.assertEqual(list(batch_iter(np.arange(2), 3, drop_last=True)),[],)

    def test_empty_and_oversized_batch(self):
        self.assertEqual(list(batch_iter([])), [])
        self.assertEqual(list(batch_iter(np.empty((0, 3)), shuffle=True)),[],)

        batches = list(batch_iter([10, 20], batch_size=5))
        self.assertEqual(len(batches), 1)
        assert_array_equal(batches[0], [10, 20])

        defaults = list(batch_iter(np.arange(33)))
        self.assertEqual([len(batch) for batch in defaults], [32, 1])

    def test_shuffle_is_reproducible_and_preserves_rows(self):
        X = np.arange(40).reshape(20, 2)
        before = X.copy()

        first = np.concatenate(list(batch_iter(X, 6, shuffle=True, random_state=42)))
        second = np.concatenate(list(batch_iter(X, 6, shuffle=True, random_state=42)))

        assert_array_equal(first, second)

        # Sorting by the unique first column recovers all original rows.
        restored = first[np.argsort(first[:, 0])]
        assert_array_equal(restored, X)

        # Both columns must move together as complete samples.
        assert_array_equal(first[:, 1], first[:, 0] + 1)
        assert_array_equal(X, before)

    def test_global_random_state_is_preserved(self):
        before = np.random.get_state()

        list(batch_iter(np.arange(20),batch_size=4,shuffle=True,random_state=42,))

        after = np.random.get_state()

        self.assertEqual(before[0], after[0])
        assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_independent_copies_and_non_contiguous_input(self):
        original = np.arange(24).reshape(6, 4)
        before = original.copy()
        X = original[:, ::2]

        self.assertFalse(X.flags.c_contiguous)

        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                batches = list(batch_iter(X,batch_size=2,shuffle=shuffle,random_state=7,))

                for batch in batches:
                    self.assertFalse(np.shares_memory(batch, original))

                self.assertFalse(np.shares_memory(batches[0], batches[1]))

                second_before = batches[1].copy()
                batches[0][:] = -999

                assert_array_equal(batches[1], second_before)
                assert_array_equal(original, before)

    def test_dtype_and_trailing_dimensions_are_preserved(self):
        inputs = [
            np.array(["a", "b", "c"]),
            np.array([True, False, True]),
            np.arange(24, dtype=np.int16).reshape(3, 2, 4),
        ]

        for X in inputs:
            with self.subTest(dtype=X.dtype):
                batches = list(batch_iter(X,batch_size=np.int64(2),random_state=np.int64(0),))

                self.assertEqual(batches[0].dtype, X.dtype)
                self.assertEqual(batches[0].shape[1:], X.shape[1:])
                assert_array_equal(np.concatenate(batches), X)

    def test_invalid_inputs_and_parameters(self):
        for value in (True, 2.0, "2"):
            with self.subTest(batch_size=value):
                with self.assertRaises(TypeError):
                    list(batch_iter([1, 2], batch_size=value))

        for value in (0, -1):
            with self.subTest(batch_size=value):
                with self.assertRaises(ValueError):
                    list(batch_iter([1, 2], batch_size=value))

        for value in (True, 1.5, "42"):
            with self.subTest(random_state=value):
                with self.assertRaises(TypeError):
                    list(batch_iter([1, 2], random_state=value))

        with self.assertRaises(ValueError):
            list(batch_iter([1, 2], random_state=-1))

        for kwargs in (
            {"shuffle": 1},
            {"shuffle": "yes"},
            {"drop_last": 0},
            {"drop_last": "no"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    list(batch_iter([1, 2], **kwargs))

        with self.assertRaises(ValueError):
            list(batch_iter(5))

        with self.assertRaises(ValueError):
            list(batch_iter([[1, 2], [3]]))

        # Empty input must still validate parameters.
        with self.assertRaises(ValueError):
            list(batch_iter([], batch_size=0))

if __name__ == "__main__":
    unittest.main()