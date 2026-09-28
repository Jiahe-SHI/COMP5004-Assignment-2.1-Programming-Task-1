import unittest
import numpy as np
from numpy.testing import assert_array_equal
from numcompute.utils import _as_float_array, _validate_integer

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

if __name__ == "__main__":
    unittest.main()