import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.utils import _as_float_array, _validate_integer, sigmoid, softmax, logsumexp

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

if __name__ == "__main__":
    unittest.main()