"""Tests for finite-difference gradients and Jacobians."""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.optim import grad, jacobian


def square_sum(v):
    return np.sum(v ** 2)


class CountingFunction:
    """Wrap f and record how many times it is called."""

    def __init__(self, f):
        self.f = f
        self.calls = 0

    def __call__(self, v):
        self.calls += 1
        return self.f(v)


class TestGrad(unittest.TestCase):

    def test_square_sum(self):
        x = np.array([1.0, -2.0, 3.0])

        for method in ("central", "forward"):
            with self.subTest(method=method):
                result = grad(square_sum, x, method=method)

                self.assertEqual(result.dtype, np.dtype("float64"))
                self.assertEqual(result.shape, (3,))
                assert_allclose(result, 2 * x, atol=1e-4)

    def test_sin_sum(self):
        x = np.array([0.0, 0.5, np.pi / 2, 2.0])

        for method in ("central", "forward"):
            with self.subTest(method=method):
                assert_allclose(grad(lambda v: np.sum(np.sin(v)), x, method=method), np.cos(x), atol=1e-4)

    def test_single_element(self):
        result = grad(lambda v: v[0] ** 3, [2.0])

        self.assertEqual(result.shape, (1,))
        assert_allclose(result, [12.0], rtol=1e-8)

    def test_python_float_output(self):
        assert_allclose(grad(lambda v: float(3 * v[0] - v[1]), [1.0, 1.0]), [3.0, -1.0], rtol=1e-8)


class TestJacobian(unittest.TestCase):

    def test_linear_square(self):
        A = np.array([[2.0, -1.0], [0.5, 3.0]])

        for method in ("central", "forward"):
            with self.subTest(method=method):
                assert_allclose(jacobian(lambda v: A @ v, [1.0, 2.0], method=method), A, atol=1e-6)

    def test_linear_non_square(self):
        A = np.arange(6.0).reshape(2, 3)
        result = jacobian(lambda v: A @ v, [1.0, -1.0, 0.5])

        self.assertEqual(result.shape, (2, 3))
        assert_allclose(result, A, atol=1e-8)

    def test_nonlinear(self):
        x = np.array([1.0, 2.0])

        def f(v):
            return np.array([v[0] * v[1], np.sin(v[0]), v[1] ** 2])

        expected = [[2.0, 1.0], [np.cos(1.0), 0.0], [0.0, 4.0]]

        assert_allclose(jacobian(f, x), expected, atol=1e-8)

    def test_single_output(self):
        x = np.array([1.0, -2.0, 3.0])
        result = jacobian(lambda v: np.array([np.sum(v ** 2)]), x)

        self.assertEqual(result.shape, (1, 3))
        assert_allclose(result, [2 * x], atol=1e-8)


class TestMethods(unittest.TestCase):

    def test_central_more_accurate_than_forward(self):
        x = np.array([0.3, 1.2])
        expected = np.exp(x)

        def f(v):
            return np.sum(np.exp(v))

        central_error = np.max(np.abs(grad(f, x, method="central", h=1e-4) - expected))
        forward_error = np.max(np.abs(grad(f, x, method="forward", h=1e-4) - expected))

        self.assertLess(forward_error, 1e-3)
        self.assertLess(central_error, 1e-7)
        self.assertLess(central_error, forward_error)

    def test_jacobian_central_more_accurate_than_forward(self):
        x = np.array([0.3, 1.2])

        def f(v):
            return np.exp(v)

        expected = np.diag(np.exp(x))
        central_error = np.max(np.abs(jacobian(f, x, method="central", h=1e-4) - expected))
        forward_error = np.max(np.abs(jacobian(f, x, method="forward", h=1e-4) - expected))

        self.assertLess(central_error, forward_error)

    def test_evaluation_counts(self):
        x = np.arange(5.0)

        for function, f in ((grad, square_sum), (jacobian, lambda v: v ** 2)):
            for method, expected in (("central", 10), ("forward", 6)):
                with self.subTest(function=function.__name__, method=method):
                    counter = CountingFunction(f)
                    function(counter, x, method=method)
                    self.assertEqual(counter.calls, expected)


class TestInputs(unittest.TestCase):

    def test_integer_and_list_input(self):
        for x in ([1, 2, 3], np.array([1, 2, 3]), (1, 2, 3)):
            with self.subTest(x=x):
                assert_allclose(grad(square_sum, x), [2.0, 4.0, 6.0], atol=1e-6)
                assert_allclose(jacobian(lambda v: 2 * v, x), 2 * np.eye(3), atol=1e-6)

    def test_non_contiguous_input(self):
        original = np.arange(10.0)
        sliced = original[::2]

        self.assertFalse(sliced.flags.c_contiguous)
        assert_allclose(grad(square_sum, sliced), 2 * sliced, atol=1e-6)
        assert_allclose(jacobian(lambda v: v ** 2, sliced), np.diag(2 * sliced), atol=1e-6)

    def test_caller_input_not_modified(self):
        x = np.array([1.0, 2.0, 3.0])

        for function, f in ((grad, square_sum), (jacobian, lambda v: v ** 2)):
            for method in ("central", "forward"):
                with self.subTest(function=function.__name__, method=method):
                    function(f, x, method=method)
                    assert_array_equal(x, [1.0, 2.0, 3.0])

    def test_f_modifying_its_argument(self):
        x = np.array([1.0, 2.0, 3.0])

        def scalar_f(v):
            value = np.sum(v ** 2)
            v[:] = 999.0
            return value

        def vector_f(v):
            value = v ** 2
            v[:] = 999.0
            return value

        for method in ("central", "forward"):
            with self.subTest(method=method):
                assert_allclose(grad(scalar_f, x, method=method), 2 * x, atol=1e-4)
                assert_allclose(jacobian(vector_f, x, method=method), np.diag(2 * x), atol=1e-4)
                assert_array_equal(x, [1.0, 2.0, 3.0])

    def test_f_receives_independent_copies(self):
        x = np.array([1.0, 2.0])
        received = []

        def f(v):
            received.append(v)
            return np.sum(v)

        grad(f, x)

        for v in received:
            self.assertFalse(np.shares_memory(v, x))


class TestErrors(unittest.TestCase):

    def assert_both_raise(self, error, f_scalar, f_vector, x, **kwargs):
        """Check that grad and jacobian raise error for both methods."""
        for method in ("central", "forward"):
            options = {"method": method, **kwargs}

            with self.subTest(function="grad", **options), self.assertRaises(error):
                grad(f_scalar, x, **options)

            with self.subTest(function="jacobian", **options), self.assertRaises(error):
                jacobian(f_vector, x, **options)

    def test_invalid_h_value(self):
        for h in (0, 0.0, -1e-5, np.inf, -np.inf, np.nan):
            with self.subTest(h=h):
                self.assert_both_raise(ValueError, square_sum, lambda v: v, [1.0], h=h)

    def test_invalid_h_type(self):
        for h in (True, False, np.bool_(True), "1e-5", None, [1e-5], 1j):
            with self.subTest(h=h):
                self.assert_both_raise(TypeError, square_sum, lambda v: v, [1.0], h=h)

    def test_invalid_x(self):
        for x in ([], np.empty(0), [[1.0, 2.0]], 1.0, [1.0, np.nan], [np.inf, 1.0], [1.0, -np.inf]):
            with self.subTest(x=x):
                self.assert_both_raise(ValueError, square_sum, lambda v: v, x)

    def test_non_numeric_x(self):
        for x in (["a", "b"], [True, False], [1, True], [1 + 2j]):
            with self.subTest(x=x):
                self.assert_both_raise(TypeError, square_sum, lambda v: v, x)

    def test_h_too_small_to_change_x(self):
        self.assert_both_raise(ValueError, square_sum, lambda v: v, [1e10, 1.0], h=1e-10)

    def test_perturbation_overflow(self):
        self.assert_both_raise(ValueError, square_sum, lambda v: v, [1e308], h=1e308)

    def test_grad_non_scalar_output(self):
        for f in (lambda v: v, lambda v: np.array([np.sum(v)]), lambda v: [1.0, 2.0]):
            for method in ("central", "forward"):
                with self.subTest(method=method), self.assertRaises(ValueError):
                    grad(f, [1.0, 2.0], method=method)

    def test_grad_non_numeric_output(self):
        for value in ("1.0", None, True, 1j):
            with self.subTest(value=value), self.assertRaises(TypeError):
                grad(lambda v: value, [1.0])

    def test_non_finite_output(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                self.assert_both_raise(ValueError, lambda v: value, lambda v: np.array([value, 1.0]), [1.0])

    def test_nan_output_on_perturbed_point_only(self):
        def f(v):
            return np.sqrt(v[0]) if v[0] >= 0 else np.nan

        with self.assertRaises(ValueError):
            grad(f, [0.0])

    def test_jacobian_wrong_dimension_output(self):
        for f in (np.sum, lambda v: np.outer(v, v), lambda v: np.empty(0)):
            for method in ("central", "forward"):
                with self.subTest(method=method), self.assertRaises(ValueError):
                    jacobian(f, [1.0, 2.0], method=method)

    def test_jacobian_non_numeric_output(self):
        with self.assertRaises(TypeError):
            jacobian(lambda v: np.array(["a", "b"]), [1.0])

    def test_jacobian_changing_length_output(self):
        for method in ("central", "forward"):
            calls = []

            def f(v):
                calls.append(None)
                return np.ones(len(calls))

            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "every evaluation"):
                jacobian(f, [1.0, 2.0], method=method)

    def test_difference_overflow(self):
        def f(v):
            return 1e308 if v[0] > 0 else -1e308

        with self.assertRaises(ValueError):
            grad(f, [0.0], h=1e-5)

    def test_errors_from_f_propagate(self):
        def f(v):
            raise KeyError("from f")

        with self.assertRaises(KeyError):
            grad(f, [1.0])


if __name__ == "__main__":
    unittest.main()
